// Package sorus reads the [models.dev] catalog of LLM providers and models,
// and exposes a provider-agnostic [Client] interface that hides each provider's
// wire-protocol details behind a single fluent [Request] builder.
//
// The typical flow is:
//
//	cat, _ := sorus.LoadCatalog(ctx)
//	client, _ := sorus.New(ctx, cat, "openrouter") // reads OPENROUTER_API_KEY from env
//	req := sorus.NewRequest(cat.Provider("openrouter").MustModel("anthropic/claude-sonnet-4-5")).
//	    System("...").
//	    Temperature(0.7).
//	    User("Hello")
//	resp, err := client.Chat(ctx, req)
//
// Supported wire protocols are listed on [New]: today, anything whose catalog
// `npm` is one of the supported values.
package sorus

import "errors"

// Errors returned by catalog and client operations.
var (
	ErrUnknownProvider     = errors.New("sorus: unknown provider id")
	ErrUnsupportedProvider = errors.New("sorus: provider npm not supported by this build")
	ErrMissingAPIKey       = errors.New("sorus: no API key found in expected env vars")
	ErrCatalogFetch        = errors.New("sorus: catalog fetch failed")
	ErrCatalogParse        = errors.New("sorus: catalog parse failed")
	ErrStreamClosed        = errors.New("sorus: stream is closed")
)
