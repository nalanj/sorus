package basis

// Provider is one catalog provider entry. It carries the metadata needed to
// pick a wire-protocol implementation and to look up credentials in the
// process environment.
type Provider struct {
	ID     string    // short id, e.g. "openrouter"
	NPM    string    // the upstream AI SDK package identifier, e.g. "@ai-sdk/openai-compatible"
	Name   string    // human-readable name, e.g. "OpenRouter"
	Env    []string  // env-var names this provider reads for credentials, in priority order
	API    string    // base URL, e.g. "https://openrouter.ai/api/v1"; empty if the SDK ships its own default
	Doc    string    // docs URL (may be empty)

	models map[string]*Model // unexported to avoid colliding with the Models() getter
}

// Model is one model record on a provider. Model is pure description — no
// configuration is attached. Configuration lives on [Request].
type Model struct {
	ID               string             // provider-local id, e.g. "anthropic/claude-sonnet-4-5"
	Name             string             // "Claude Sonnet 4.5"
	Family           string             // "claude-sonnet"
	Description      string
	Modalities       ModelModalities    // {input: [...], output: [...]}
	Limit            ModelLimit         // {context, input, output}
	Cost             ModelCost          // per-1M-token USD prices; zero means "not advertised"
	Reasoning        bool
	ReasoningOptions []ReasoningOption  // effort enum / toggle / budget_tokens
	ToolCall         bool
	StructuredOutput bool
	Temperature      bool
	Attachment       bool
	Knowledge        string             // cutoff, "YYYY-MM" or "YYYY-MM-DD"
	ReleaseDate      string             // "YYYY-MM-DD"
	LastUpdated      string             // "YYYY-MM-DD"
	Status           string             // "", "alpha", "beta", "deprecated"
	OpenWeights      bool

	provider *Provider // back-reference, set at catalog-load time
}

type ModelModalities struct {
	Input  []string
	Output []string
}

type ModelLimit struct {
	Context int
	Input   int
	Output  int
}

type ModelCost struct {
	Input       float64
	Output      float64
	CacheRead   float64
	CacheWrite  float64
	Reasoning   float64
	InputAudio  float64
	OutputAudio float64
}

// ReasoningOptionType is the shape of one entry in [Model.ReasoningOptions].
type ReasoningOptionType string

const (
	ReasoningEffort       ReasoningOptionType = "effort"
	ReasoningToggle       ReasoningOptionType = "toggle"
	ReasoningBudgetTokens ReasoningOptionType = "budget_tokens"
)

// ReasoningOption describes a model's reasoning control surface.
type ReasoningOption struct {
	Type   ReasoningOptionType
	Values []string // populated when Type == ReasoningEffort
	Min    *int     // populated when Type == ReasoningBudgetTokens
	Max    *int     // populated when Type == ReasoningBudgetTokens (may be nil)
}

// Ref returns the provider-local id. Useful for logging.
func (m *Model) Ref() string { return m.ID }

// Provider returns the parent provider of this model.
func (m *Model) Provider() *Provider { return m.provider }

// Model returns the model with the given id, or nil if unknown.
func (p *Provider) Model(id string) (*Model, bool) {
	m, ok := p.models[id]
	return m, ok
}

// MustModel returns the model with the given id, panicking if unknown.
func (p *Provider) MustModel(id string) *Model {
	m, ok := p.models[id]
	if !ok {
		panic("basis: provider " + p.ID + " has no model " + id)
	}
	return m
}

// Models returns all models for this provider.
func (p *Provider) Models() []*Model {
	out := make([]*Model, 0, len(p.models))
	for _, m := range p.models {
		out = append(out, m)
	}
	return out
}

// ModelsByFamily returns models on this provider whose Family matches.
func (p *Provider) ModelsByFamily(family string) []*Model {
	var out []*Model
	for _, m := range p.models {
		if m.Family == family {
			out = append(out, m)
		}
	}
	return out
}

// Catalog is the loaded models.dev catalog.
type Catalog struct {
	providers map[string]*Provider
}

// Provider returns the provider with the given id, or nil if unknown.
func (c *Catalog) Provider(id string) *Provider {
	return c.providers[id]
}

// MustProvider returns the provider with the given id, panicking if unknown.
func (c *Catalog) MustProvider(id string) *Provider {
	p, ok := c.providers[id]
	if !ok {
		panic("basis: catalog has no provider " + id)
	}
	return p
}

// Providers returns all providers.
func (c *Catalog) Providers() []*Provider {
	out := make([]*Provider, 0, len(c.providers))
	for _, p := range c.providers {
		out = append(out, p)
	}
	return out
}

// Models returns every model across every provider (cross-provider enumeration).
func (c *Catalog) Models() []*Model {
	var out []*Model
	for _, p := range c.providers {
		for _, m := range p.models {
			out = append(out, m)
		}
	}
	return out
}

// ModelsByFamily returns every model across every provider whose Family matches.
func (c *Catalog) ModelsByFamily(family string) []*Model {
	var out []*Model
	for _, p := range c.providers {
		for _, m := range p.models {
			if m.Family == family {
				out = append(out, m)
			}
		}
	}
	return out
}

// FindModel looks up a model by qualified id of the form "provider/model".
// Returns the provider, the model, and ok.
func (c *Catalog) FindModel(qualifiedID string) (*Provider, *Model, bool) {
	for i := 0; i < len(qualifiedID); i++ {
		if qualifiedID[i] == '/' {
			pid := qualifiedID[:i]
			mid := qualifiedID[i+1:]
			p := c.Provider(pid)
			if p == nil {
				return nil, nil, false
			}
			m, ok := p.Model(mid)
			if !ok {
				return p, nil, false
			}
			return p, m, true
		}
	}
	return nil, nil, false
}
