package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/routatic/proxy/internal/config"
)

func newAnthropicFirstTestHandler(baseURL string, enabled bool, fallback http.Handler) *AnthropicFirstHandler {
	cfg := &config.Config{AnthropicFirst: config.AnthropicFirstConfig{Enabled: enabled, BaseURL: baseURL}}
	return NewAnthropicFirstHandler(config.NewAtomicConfig(cfg, "/tmp/test-config.json"), fallback)
}

func TestAnthropicFirstHealthyPassthrough(t *testing.T) {
	var gotBody, gotBeta, gotAuth, gotPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		gotBeta = r.Header.Get("anthropic-beta")
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.RequestURI()
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Upstream", "anthropic")
		w.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(w, "event: message_stop\ndata: {}\n\n")
	}))
	defer upstream.Close()

	var fallbackCalls atomic.Int32
	h := newAnthropicFirstTestHandler(upstream.URL, true, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		fallbackCalls.Add(1)
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages?beta=true", strings.NewReader(`{"model":"claude-sonnet-4-6"}`))
	req.Header.Set("Authorization", "Bearer oauth-token")
	req.Header.Set("anthropic-beta", "oauth-2025-04-20,context-management-2025-06-27")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated || rec.Header().Get("X-Upstream") != "anthropic" {
		t.Fatalf("response = %d %v, want Anthropic 201", rec.Code, rec.Header())
	}
	if gotBody != `{"model":"claude-sonnet-4-6"}` || gotAuth != "Bearer oauth-token" {
		t.Fatalf("body/auth not forwarded: body=%q auth=%q", gotBody, gotAuth)
	}
	if gotBeta == "" || gotPath != "/v1/messages?beta=true" {
		t.Fatalf("beta/path not forwarded: beta=%q path=%q", gotBeta, gotPath)
	}
	if fallbackCalls.Load() != 0 {
		t.Fatal("fallback called for healthy Anthropic response")
	}
}

func TestAnthropicFirstFallsBackOnAvailabilityFailures(t *testing.T) {
	for _, status := range []int{http.StatusRequestTimeout, http.StatusTooManyRequests, 500, 503, 529} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Retry-After", "120")
				http.Error(w, "unavailable", status)
			}))
			defer upstream.Close()

			var gotBody string
			h := newAnthropicFirstTestHandler(upstream.URL, true, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				gotBody = string(body)
				w.Header().Set("X-Upstream", "opencode")
				w.WriteHeader(http.StatusOK)
			}))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("request-body")))

			if rec.Code != http.StatusOK || rec.Header().Get("X-Upstream") != "opencode" || gotBody != "request-body" {
				t.Fatalf("fallback response=%d header=%q body=%q", rec.Code, rec.Header().Get("X-Upstream"), gotBody)
			}
		})
	}
}

func TestAnthropicFirstPreservesClientErrors(t *testing.T) {
	for _, status := range []int{400, 401, 403, 404, 422} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("X-Error", "original")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, `{"type":"error","error":{"message":"original"}}`)
			}))
			defer upstream.Close()

			var fallbackCalls atomic.Int32
			h := newAnthropicFirstTestHandler(upstream.URL, true, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				fallbackCalls.Add(1)
			}))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}")))

			if rec.Code != status || rec.Header().Get("X-Error") != "original" || !strings.Contains(rec.Body.String(), "original") {
				t.Fatalf("response was not preserved: status=%d headers=%v body=%q", rec.Code, rec.Header(), rec.Body.String())
			}
			if fallbackCalls.Load() != 0 {
				t.Fatal("fallback called for client error")
			}
		})
	}
}

func TestAnthropicFirstDisabledUsesFallback(t *testing.T) {
	var calls atomic.Int32
	h := newAnthropicFirstTestHandler("https://api.anthropic.com", false, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}")))
	if calls.Load() != 1 || rec.Code != http.StatusNoContent {
		t.Fatalf("disabled mode calls=%d status=%d", calls.Load(), rec.Code)
	}
}

func TestAnthropicFirstTransportFailureFallsBack(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	baseURL := upstream.URL
	upstream.Close()

	var calls atomic.Int32
	h := newAnthropicFirstTestHandler(baseURL, true, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}")))
	if calls.Load() != 1 || rec.Code != http.StatusNoContent {
		t.Fatalf("transport fallback calls=%d status=%d", calls.Load(), rec.Code)
	}
}

func TestAnthropicFirstDoesNotFollowRedirectWithCredentials(t *testing.T) {
	var redirectedCalls atomic.Int32
	destination := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirectedCalls.Add(1)
	}))
	defer destination.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", destination.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	h := newAnthropicFirstTestHandler(source.URL, true, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("fallback called for redirect")
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}")))
	if rec.Code != http.StatusTemporaryRedirect || redirectedCalls.Load() != 0 {
		t.Fatalf("status=%d redirectedCalls=%d", rec.Code, redirectedCalls.Load())
	}
}

func TestAvailabilityGateBackoffAndSingleProbe(t *testing.T) {
	var gate availabilityGate
	now := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	attempt, ok := gate.allow(now, "https://api.anthropic.com")
	if !ok || attempt.probe {
		t.Fatal("healthy gate should allow a normal request")
	}
	if delay := gate.failed(now, attempt, ""); delay != 30*time.Second {
		t.Fatalf("first backoff=%v, want 30s", delay)
	}
	if _, ok := gate.allow(now.Add(29*time.Second), "https://api.anthropic.com"); ok {
		t.Fatal("request allowed before backoff elapsed")
	}
	probe, ok := gate.allow(now.Add(30*time.Second), "https://api.anthropic.com")
	if !ok || !probe.probe {
		t.Fatal("expected half-open probe")
	}
	if _, ok := gate.allow(now.Add(30*time.Second), "https://api.anthropic.com"); ok {
		t.Fatal("concurrent half-open probe allowed")
	}
	if delay := gate.failed(now.Add(30*time.Second), probe, "120"); delay != 2*time.Minute {
		t.Fatalf("Retry-After delay=%v, want 2m", delay)
	}
	probe, ok = gate.allow(now.Add(150*time.Second), "https://api.anthropic.com")
	if !ok || !probe.probe {
		t.Fatal("expected second half-open probe")
	}
	gate.available(probe)
	if attempt, ok = gate.allow(now.Add(150*time.Second), "https://api.anthropic.com"); !ok || attempt.probe {
		t.Fatal("successful probe did not close gate")
	}
}

func TestAvailabilityGateConcurrentProbe(t *testing.T) {
	var gate availabilityGate
	now := time.Now()
	attempt, _ := gate.allow(now, "upstream")
	gate.failed(now, attempt, "0")

	// Retry-After: 0 means the probe window opens immediately.
	probeTime := now.Add(50 * time.Millisecond)

	var allowed atomic.Int32
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, ok := gate.allow(probeTime, "upstream"); ok {
				allowed.Add(1)
			}
		}()
	}
	wg.Wait()
	if allowed.Load() != 1 {
		t.Fatalf("allowed %d probes, want 1", allowed.Load())
	}
}

func TestParseRetryAfterHTTPDate(t *testing.T) {
	now := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	delay, ok := parseRetryAfter(now.Add(90*time.Second).Format(http.TimeFormat), now)
	if !ok || delay != 90*time.Second {
		t.Fatalf("delay=%v ok=%v, want 90s", delay, ok)
	}
}
