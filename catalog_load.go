package basis

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"sync"
)

// defaultCatalogURL is the upstream models.dev JSON endpoint.
const defaultCatalogURL = "https://models.dev/api.json"

// CatalogOption configures [LoadCatalog].
type CatalogOption func(*catalogConfig)

type catalogConfig struct {
	refresh    bool
	catalogURL string
	httpClient *http.Client
}

// WithRefresh forces a fresh fetch even when an in-memory cache exists.
func WithRefresh() CatalogOption {
	return func(c *catalogConfig) { c.refresh = true }
}

// WithCatalogURL overrides the URL to fetch the catalog from. The default
// is the upstream https://models.dev/api.json. Any non-default URL bypasses
// the in-memory cache and re-fetches on every call.
//
// Use a file:// URI for a vendored copy.
func WithCatalogURL(url string) CatalogOption {
	return func(c *catalogConfig) { c.catalogURL = url }
}

// WithFetchHTTPClient lets callers inject a custom http.Client used for
// fetching the catalog (timeouts, retries, transport instrumentation).
// Supplying one bypasses the in-memory cache so the custom client is
// actually used.
func WithFetchHTTPClient(h *http.Client) CatalogOption {
	return func(c *catalogConfig) { c.httpClient = h }
}

func defaultCatalogConfig() catalogConfig {
	return catalogConfig{catalogURL: defaultCatalogURL}
}

// In-memory catalog cache. Populated on the first [LoadCatalog] call with
// default options; cleared by [WithRefresh] or any non-default url/client.
var (
	catalogCacheMu sync.RWMutex
	catalogCache   *Catalog
)

// LoadCatalog fetches the models.dev catalog and parses it. The result is
// cached in memory for the lifetime of the process; subsequent calls with
// default options return the cached instance.
//
// [LoadCatalogFromBytes] and [LoadCatalogFromFS] bypass the cache because
// they take user-supplied data and never touch the network.
func LoadCatalog(ctx context.Context, opts ...CatalogOption) (*Catalog, error) {
	cfg := defaultCatalogConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	bypassCache := cfg.refresh ||
		cfg.catalogURL != defaultCatalogURL ||
		cfg.httpClient != nil

	if !bypassCache {
		catalogCacheMu.RLock()
		cat := catalogCache
		catalogCacheMu.RUnlock()
		if cat != nil {
			return cat, nil
		}
	}

	data, err := fetchCatalog(ctx, cfg.catalogURL, cfg.httpClient)
	if err != nil {
		return nil, err
	}
	cat, err := parseCatalog(data)
	if err != nil {
		return nil, err
	}

	if !bypassCache {
		catalogCacheMu.Lock()
		catalogCache = cat
		catalogCacheMu.Unlock()
	}
	return cat, nil
}

// LoadCatalogFromBytes parses an in-memory catalog JSON blob.
func LoadCatalogFromBytes(b []byte) (*Catalog, error) {
	return parseCatalog(b)
}

// LoadCatalogFromFS parses a catalog stored in the given fs.FS.
func LoadCatalogFromFS(fsys fs.FS, name string) (*Catalog, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCatalogFetch, err)
	}
	return parseCatalog(data)
}

func fetchCatalog(ctx context.Context, url string, hc *http.Client) ([]byte, error) {
	if url == "" {
		return nil, fmt.Errorf("%w: empty catalog URL", ErrCatalogFetch)
	}
	if len(url) > 7 && url[:7] == "file://" {
		data, err := os.ReadFile(url[7:])
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCatalogFetch, err)
		}
		return data, nil
	}

	client := hc
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCatalogFetch, err)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCatalogFetch, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: status %d", ErrCatalogFetch, resp.StatusCode)
	}
	// Cap read at 32 MiB so a misbehaving upstream can't exhaust memory.
	data, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCatalogFetch, err)
	}
	return data, nil
}

// --- raw json schema ---

type rawProvider struct {
	ID     string               `json:"id"`
	NPM    string               `json:"npm"`
	Name   string               `json:"name"`
	Env    []string             `json:"env"`
	API    string               `json:"api,omitempty"`
	Doc    string               `json:"doc,omitempty"`
	Models map[string]*rawModel `json:"models"`
}

type rawModel struct {
	ID               string               `json:"id"`
	Name             string               `json:"name"`
	Family           string               `json:"family"`
	Description      string               `json:"description"`
	Modalities       *rawModalities       `json:"modalities"`
	Limit            *rawLimit            `json:"limit"`
	Cost             *rawCost             `json:"cost"`
	Reasoning        bool                 `json:"reasoning"`
	ReasoningOptions []rawReasoningOption `json:"reasoning_options"`
	ToolCall         bool                 `json:"tool_call"`
	StructuredOutput bool                 `json:"structured_output"`
	Temperature      bool                 `json:"temperature"`
	Attachment       bool                 `json:"attachment"`
	Knowledge        string               `json:"knowledge"`
	ReleaseDate      string               `json:"release_date"`
	LastUpdated      string               `json:"last_updated"`
	Status           string               `json:"status"`
	OpenWeights      bool                 `json:"open_weights"`
}

type rawModalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

type rawLimit struct {
	Context int `json:"context"`
	Input   int `json:"input"`
	Output  int `json:"output"`
}

type rawCost struct {
	Input       float64 `json:"input"`
	Output      float64 `json:"output"`
	CacheRead   float64 `json:"cache_read"`
	CacheWrite  float64 `json:"cache_write"`
	Reasoning   float64 `json:"reasoning"`
	InputAudio  float64 `json:"input_audio"`
	OutputAudio float64 `json:"output_audio"`
}

type rawReasoningOption struct {
	Type   ReasoningOptionType `json:"type"`
	Values []string            `json:"values"`
	Min    *int                `json:"min"`
	Max    *int                `json:"max"`
}

func parseCatalog(data []byte) (*Catalog, error) {
	var raw map[string]*rawProvider
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCatalogParse, err)
	}

	c := &Catalog{providers: make(map[string]*Provider, len(raw))}
	for pid, rp := range raw {
		if rp == nil {
			continue
		}
		p := &Provider{
			ID:     firstNonEmpty(rp.ID, pid),
			NPM:    rp.NPM,
			Name:   rp.Name,
			Env:    rp.Env,
			API:    rp.API,
			Doc:    rp.Doc,
			models: make(map[string]*Model, len(rp.Models)),
		}
		for mid, rm := range rp.Models {
			if rm == nil {
				continue
			}
			m := convertModel(rm)
			m.provider = p
			p.models[firstNonEmpty(rm.ID, mid)] = m
		}
		c.providers[p.ID] = p
	}
	return c, nil
}

func convertModel(rm *rawModel) *Model {
	m := &Model{
		ID:               rm.ID,
		Name:             rm.Name,
		Family:           rm.Family,
		Description:      rm.Description,
		Reasoning:        rm.Reasoning,
		ToolCall:         rm.ToolCall,
		StructuredOutput: rm.StructuredOutput,
		Temperature:      rm.Temperature,
		Attachment:       rm.Attachment,
		Knowledge:        rm.Knowledge,
		ReleaseDate:      rm.ReleaseDate,
		LastUpdated:      rm.LastUpdated,
		Status:           rm.Status,
		OpenWeights:      rm.OpenWeights,
	}
	if rm.Modalities != nil {
		m.Modalities = ModelModalities{
			Input:  rm.Modalities.Input,
			Output: rm.Modalities.Output,
		}
	}
	if rm.Limit != nil {
		m.Limit = ModelLimit{
			Context: rm.Limit.Context,
			Input:   rm.Limit.Input,
			Output:  rm.Limit.Output,
		}
	}
	if rm.Cost != nil {
		m.Cost = ModelCost{
			Input:       rm.Cost.Input,
			Output:      rm.Cost.Output,
			CacheRead:   rm.Cost.CacheRead,
			CacheWrite:  rm.Cost.CacheWrite,
			Reasoning:   rm.Cost.Reasoning,
			InputAudio:  rm.Cost.InputAudio,
			OutputAudio: rm.Cost.OutputAudio,
		}
	}
	for _, ro := range rm.ReasoningOptions {
		m.ReasoningOptions = append(m.ReasoningOptions, ReasoningOption{
			Type:   ro.Type,
			Values: ro.Values,
			Min:    ro.Min,
			Max:    ro.Max,
		})
	}
	return m
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
