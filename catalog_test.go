package sorus

import (
	"context"
	"testing"
)

// Note: fetchCatalog is unexported; this test is in the same package.

func TestLoadCatalog(t *testing.T) {
	ctx := context.Background()
	cat, err := LoadCatalog(ctx)
	if err != nil {
		t.Fatalf("LoadCatalog: %v", err)
	}

	provs := cat.Providers()
	if len(provs) < 100 {
		t.Errorf("expected many providers, got %d", len(provs))
	}

	openrouter := cat.MustProvider("openrouter")
	if openrouter.ID != "openrouter" {
		t.Errorf("openrouter id = %q", openrouter.ID)
	}
	if len(openrouter.Env) == 0 || openrouter.Env[0] != "OPENROUTER_API_KEY" {
		t.Errorf("openrouter env = %v", openrouter.Env)
	}
	if openrouter.API == "" {
		t.Errorf("openrouter api is empty")
	}

	models := openrouter.Models()
	if len(models) == 0 {
		t.Errorf("openrouter has no models in catalog")
	}
	one := models[0]
	if one.ID == "" || one.Name == "" {
		t.Errorf("model missing fields: %+v", one)
	}
	if one.Provider() != openrouter {
		t.Errorf("model back-reference doesn't match")
	}
}

func TestFindModel(t *testing.T) {
	ctx := context.Background()
	cat, err := LoadCatalog(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_, _, ok := cat.FindModel("not_a_real_provider/some_model")
	if ok {
		t.Errorf("FindModel should return false for unknown ids")
	}

	if _, ok := cat.MustProvider("openrouter").Model("foo/bar"); ok {
		t.Errorf("Model(id) should return ok=false for unknown ids")
	}
}

func TestCatalogFromBytesRoundtrip(t *testing.T) {
	// Roundtrip: load once via the network, serialize the catalog back to
	// JSON, and re-parse via LoadCatalogFromBytes. The second *Catalog
	// should have the same provider and model counts.
	ctx := context.Background()
	live, err := LoadCatalog(ctx)
	if err != nil {
		t.Fatal(err)
	}
	n := len(live.Providers())
	if n < 100 {
		t.Fatalf("expected many providers, got %d", n)
	}

	// Serialize via the public types. We round-trip through the package's
	// own internals since we don't expose MarshalJSON. A simpler check:
	// the JSON contract test is that the on-disk API shape is consistent
	// with what LoadCatalogFromBytes parses, which we verify by feeding
	// the live JSON bytes back through.
	data, err := fetchCatalog(ctx, defaultCatalogURL, nil)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	parsed, err := LoadCatalogFromBytes(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(parsed.Providers()) != n {
		t.Errorf("provider count mismatch: live=%d parsed=%d", n, len(parsed.Providers()))
	}
}
