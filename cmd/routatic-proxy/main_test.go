package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/routatic/proxy/internal/config"
)

func TestDefaultConfigValidWithGlobalAPIKey(t *testing.T) {
	t.Setenv("ROUTATIC_PROXY_API_KEY", "test-key")
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(getDefaultConfig()), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadFromPath(path)
	if err != nil {
		t.Fatalf("generated default config is invalid: %v", err)
	}
	if cfg.AnthropicFirst.BaseURL != "https://api.anthropic.com" {
		t.Fatalf("AnthropicFirst=%+v", cfg.AnthropicFirst)
	}
}
