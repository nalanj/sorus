package basis

import (
	"context"
	"fmt"
	"net/http"
	"os"
)

// Client is the provider-agnostic chat surface. Implementations are
// selected by [New] based on the provider's `npm` value in the catalog.
//
// A Client is bound to one provider. To switch providers, call [New] again.
type Client interface {
	// Chat sends req and returns a single non-streaming [Response].
	Chat(ctx context.Context, req *Request) (*Response, error)

	// Stream sends req and returns a streaming [Stream]. Always call
	// [Stream.Close] when done, even on error paths.
	Stream(ctx context.Context, req *Request) (*Stream, error)
}

// Option configures [New].
type Option func(*config)

type config struct {
	apiKey     string
	apiKeySet  bool
	httpClient *http.Client
}

// WithAPIKey overrides the API-key lookup. By default the catalog's `env`
// list is consulted for an env-var that's set.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey = key
		c.apiKeySet = true
	}
}

// WithHTTPClient injects a custom *http.Client used by the underlying
// provider transport.
func WithHTTPClient(h *http.Client) Option {
	return func(c *config) { c.httpClient = h }
}

// New builds a [Client] for the named provider.
//
// The provider's `env` list determines API-key resolution: the first name
// that's set in the process environment wins. Override with [WithAPIKey]
// if you already have a credential in hand (tests, multi-tenant apps,
// key rings).
//
// Returns [ErrUnknownProvider] if the provider id isn't in the catalog,
// [ErrUnsupportedProvider] if the provider's `npm` value isn't handled by
// this build (today: anything other than `@ai-sdk/openai-compatible` or
// `@ai-sdk/openai`), or [ErrMissingAPIKey] if no credential was found and
// no [WithAPIKey] was supplied.
func New(ctx context.Context, cat *Catalog, providerID string, opts ...Option) (Client, error) {
	p := cat.Provider(providerID)
	if p == nil {
		return nil, fmt.Errorf("%w: %q", ErrUnknownProvider, providerID)
	}

	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}

	if !cfg.apiKeySet {
		var found bool
		for _, envName := range p.Env {
			if v, ok := os.LookupEnv(envName); ok && v != "" {
				cfg.apiKey = v
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("%w: provider %s expects one of %v", ErrMissingAPIKey, p.ID, p.Env)
		}
	}

	switch p.NPM {
	case "@ai-sdk/openai-compatible", "@ai-sdk/openai":
		return newChatCompletionsClient(p, cfg.apiKey, cfg.httpClient)
	case "@ai-sdk/anthropic":
		return newAnthropicClient(p, cfg.apiKey, cfg.httpClient)
	default:
		return nil, fmt.Errorf("%w: provider %s (npm=%q)", ErrUnsupportedProvider, p.ID, p.NPM)
	}
}
