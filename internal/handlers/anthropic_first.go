package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/routatic/proxy/internal/config"
)

const (
	availabilityBackoffStart = 30 * time.Second
	availabilityBackoffMax   = 15 * time.Minute
	maxAnthropicBodySize     = 100 << 20
)

// AnthropicFirstHandler sends inference to Anthropic and uses the existing
// OpenCode handler only while Anthropic is unavailable.
type AnthropicFirstHandler struct {
	atomic   *config.AtomicConfig
	fallback http.Handler
	client   *http.Client
	logger   *slog.Logger
	gate     availabilityGate
}

type availabilityGate struct {
	mu          sync.Mutex
	baseURL     string
	unavailable bool
	probing     bool
	failures    int
	nextProbe   time.Time
}

type availabilityAttempt struct {
	baseURL string
	probe   bool
}

func (g *availabilityGate) allow(now time.Time, baseURL string) (availabilityAttempt, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.baseURL != baseURL {
		g.resetLocked(baseURL)
	}
	if !g.unavailable {
		return availabilityAttempt{baseURL: baseURL}, true
	}
	if now.Before(g.nextProbe) || g.probing {
		return availabilityAttempt{}, false
	}
	g.probing = true
	return availabilityAttempt{baseURL: baseURL, probe: true}, true
}

func (g *availabilityGate) failed(now time.Time, attempt availabilityAttempt, retryAfter string) time.Duration {
	g.mu.Lock()
	defer g.mu.Unlock()
	if attempt.baseURL != g.baseURL {
		return 0
	}

	g.unavailable = true
	if attempt.probe {
		g.probing = false
	}
	g.failures++
	delay, ok := parseRetryAfter(retryAfter, now)
	if !ok {
		delay = availabilityBackoffStart
		for i := 1; i < g.failures && delay < availabilityBackoffMax; i++ {
			delay *= 2
		}
		if delay > availabilityBackoffMax {
			delay = availabilityBackoffMax
		}
	}
	g.nextProbe = now.Add(delay)
	return delay
}

func (g *availabilityGate) available(attempt availabilityAttempt) {
	if !attempt.probe {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if attempt.baseURL != g.baseURL {
		return
	}
	g.resetLocked(g.baseURL)
}

func (g *availabilityGate) reset(baseURL string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.resetLocked(baseURL)
}

func (g *availabilityGate) resetLocked(baseURL string) {
	g.baseURL = baseURL
	g.unavailable = false
	g.probing = false
	g.failures = 0
	g.nextProbe = time.Time{}
}

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	if seconds, err := strconv.Atoi(strings.TrimSpace(value)); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	if when.Before(now) {
		return 0, true
	}
	return when.Sub(now), true
}

// NewAnthropicFirstHandler creates the opt-in Anthropic passthrough layer.
func NewAnthropicFirstHandler(atomic *config.AtomicConfig, fallback http.Handler) *AnthropicFirstHandler {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       90 * time.Second,
	}
	return &AnthropicFirstHandler{
		atomic:   atomic,
		fallback: fallback,
		client: &http.Client{
			Transport: transport,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		logger: slog.Default(),
	}
}

func (h *AnthropicFirstHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cfg := h.atomic.Get().AnthropicFirst
	if !cfg.Enabled || r.Method != http.MethodPost {
		if !cfg.Enabled {
			h.gate.reset(cfg.BaseURL)
		}
		h.fallback.ServeHTTP(w, r)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxAnthropicBodySize+1))
	if err != nil {
		h.writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	if len(body) > maxAnthropicBodySize {
		h.writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
		return
	}

	attempt, allowed := h.gate.allow(time.Now(), cfg.BaseURL)
	if !allowed {
		h.serveFallback(w, r, body)
		return
	}

	upstreamReq, err := newAnthropicRequest(r, cfg.BaseURL, body)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "invalid Anthropic base URL")
		return
	}
	resp, err := h.client.Do(upstreamReq)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		delay := h.gate.failed(time.Now(), attempt, "")
		h.logger.Warn("Anthropic unavailable, using OpenCode", "error", err, "retry_in", delay)
		h.serveFallback(w, r, body)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if isAnthropicAvailabilityFailure(resp.StatusCode) {
		delay := h.gate.failed(time.Now(), attempt, resp.Header.Get("Retry-After"))
		h.logger.Warn("Anthropic unavailable, using OpenCode", "status", resp.StatusCode, "retry_in", delay)
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		h.serveFallback(w, r, body)
		return
	}

	h.gate.available(attempt)
	h.logger.Debug("Anthropic request succeeded", "status", resp.StatusCode)
	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	copyStreamingResponse(r.Context(), w, resp.Body)
}

func newAnthropicRequest(in *http.Request, baseURL string, body []byte) (*http.Request, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	base.Path = strings.TrimRight(base.Path, "/") + in.URL.Path
	base.RawQuery = in.URL.RawQuery
	out, err := http.NewRequestWithContext(in.Context(), in.Method, base.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	out.Header = in.Header.Clone()
	removeHopHeaders(out.Header)
	out.ContentLength = int64(len(body))
	return out, nil
}

func isAnthropicAvailabilityFailure(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500
}

func (h *AnthropicFirstHandler) serveFallback(w http.ResponseWriter, r *http.Request, body []byte) {
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	h.fallback.ServeHTTP(w, r)
}

func copyHeader(dst, src http.Header) {
	dstClone := src.Clone()
	removeHopHeaders(dstClone)
	for key, values := range dstClone {
		dst[key] = append([]string(nil), values...)
	}
}

func removeHopHeaders(header http.Header) {
	for _, value := range header.Values("Connection") {
		for _, key := range strings.Split(value, ",") {
			header.Del(strings.TrimSpace(key))
		}
	}
	for _, key := range []string{
		"Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate",
		"Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade",
	} {
		header.Del(key)
	}
}

var streamBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 32<<10)
		return &b
	},
}

func copyStreamingResponse(ctx context.Context, w http.ResponseWriter, body io.Reader) {
	bufPtr := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(bufPtr)
	buf := *bufPtr
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		n, err := body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

func (h *AnthropicFirstHandler) writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type":  "error",
		"error": map[string]string{"type": "invalid_request_error", "message": message},
	})
}
