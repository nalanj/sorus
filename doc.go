// Package basis reads the [models.dev] catalog of LLM providers and models,
// and exposes a provider-agnostic [Client] interface that hides each provider's
// wire-protocol details behind a single fluent [Request] builder.
//
// The typical flow is:
//
//	cat, _ := basis.LoadCatalog(ctx)
//	client, _ := basis.New(ctx, cat, "openrouter") // reads OPENROUTER_API_KEY from env
//	req := basis.NewRequest(client.Model("anthropic", "claude-sonnet-4-5")).
//	    System("...").
//	    Temperature(0.7).
//	    User("Hello")
//	resp, err := client.Chat(ctx, req)
//
// Today the package supports providers with `npm: "@ai-sdk/openai-compatible"`
// or `npm: "@ai-sdk/openai"` in their catalog entry — see [New]. Future
// implementations plug into the same [Client] surface.
package basis

import "errors"

// Errors returned by catalog and client operations.
var (
	ErrUnknownProvider     = errors.New("basis: unknown provider id")
	ErrUnsupportedProvider = errors.New("basis: provider npm not supported by this build")
	ErrMissingAPIKey       = errors.New("basis: no API key found in expected env vars")
	ErrCatalogFetch        = errors.New("basis: catalog fetch failed")
	ErrCatalogParse        = errors.New("basis: catalog parse failed")
	ErrStreamClosed        = errors.New("basis: stream is closed")
)
