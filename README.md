# sorus

> A small Go package that reads the [models.dev](https://models.dev) catalog and exposes a provider-agnostic interface for calling the models on it.

The package does two things:

1. Loads the [models.dev](https://models.dev) catalog of **222 providers** and ~6,000 model records into typed Go structs, with lookup helpers for cross-provider queries.
2. Hands back a `Client` interface for any provider in the catalog. `Request` is built by fluent configuration starting with `NewRequest(model, ...)`; chat-completions, Anthropic, Bedrock, etc. all plug into the same `Client`.

Today one wire protocol ships; future implementations plug into the same surface.

## Status

**MVP in progress.** v1 ships:

- Catalog loader (in-memory cache + vendor-friendly loaders).
- Catalog types + lookup helpers.
- Fluent `Request` builder starting with `NewRequest(model, ...)`.
- Generic `Client`, `Response`, `Stream`, `Event` types.
- `Client` implementation for `@ai-sdk/openai-compatible` and `@ai-sdk/openai` providers (~187 of the 222 providers in the catalog).

Deferred:

- Anthropic / Bedrock / Vertex / Gemini native implementations (one wire protocol at a time, all behind the same `Client`).
- Embeddings, image generation, audio.
- Per-provider reasoning mapping.

## Install

```bash
go get github.com/nalanj/sorus
```

Requires Go 1.22+. Wire-protocol handling is internal — users don't need any provider SDKs in their project.

> **Import alias.** Because the package name is `sorus`, feel free to alias on import if you'd prefer:
> ```go
> import m "github.com/nalanj/sorus"
> ```
> The README examples show unaliased usage; substitute your alias if it fits.

## Quick start

```go
package main

import (
    "context"
    "fmt"

    "github.com/nalanj/sorus"
)

func main() {
    ctx := context.Background()

    // 1. Load the catalog. Fetches https://models.dev/api.json and caches it
    //    under the OS user cache dir. ~400 KB JSON.
    cat, err := sorus.LoadCatalog(ctx)
    if err != nil {
        panic(err)
    }

    // 2. Build a client for a provider. New() looks up OPENROUTER_API_KEY in
    //    os.Getenv (catalog says that's the right name for OpenRouter), errors
    //    if missing, and hands back a *sorus.Client with the right base URL.
    client, err := sorus.New(ctx, cat, "openrouter")
    if err != nil {
        panic(err)
    }

    // 3. Find a model. Model is pure description — no configuration attached.
    model := cat.Provider("openrouter").MustModel("anthropic/claude-sonnet-4-5")
    fmt.Printf("name=%s context=%d input_cost=$%f/Mtok\n",
        model.Name, model.Limit.Context, model.Cost.Input)

    // 4. Build a Request by fluent configuration. Literals go directly into
    //    chained methods — no Ptr helper to learn.
    req := sorus.NewRequest(model).
        System("You are a helpful assistant.").
        Temperature(0.7).
        MaxTokens(2048).
        User("Hello, world!")

    resp, err := client.Chat(ctx, req)
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Message.Text())
}
```

## Design decisions

These are the choices you should push back on if they don't match what you want, before we write the implementation.

### Wrapped, not pass-through

The catalog says ~187 of 222 providers speak OpenAI's chat completions protocol (with `npm: "@ai-sdk/openai-compatible"` or `npm: "@ai-sdk/openai"`). But "OpenAI-compatible" is the wire, not the API. Different providers expose different shapes (one has `reasoning_effort`, another has `enable_thinking`, a third has neither). The point of this package is to *hide* those differences behind one type — so when we add Anthropic, Bedrock, Vertex, the call site doesn't change.

So we wrap. Users import only `github.com/nalanj/sorus`. The package internally implements the `Client` interface for each supported wire protocol — one today, more later — and converts between the fluent `Request` shape and each provider's wire format.

There is no protocol-specific escape hatch by design. If the `Client` interface misses something you need, the wrapper is the seam — file an issue. We deliberately don't expose wire-protocol SDK types from the package's public surface.

### Configuration lives on `Request`, not on `Model`

`Model` is pure description: name, family, modalities, limits, cost, capabilities, release date. Nothing on it can be configured.

`Request` is fluent configuration: you call `NewRequest(model, ...)` and chain methods like `.Temperature(0.7)`, `.Tools(...)`, `.User("...")`. Each method takes a literal value and stores it internally; the package owns the optional-pointer mechanics.

This keeps `Model` a thin description, makes each request self-contained, and avoids exposing any `Ptr`-style helper. Reusable defaults work via `Request.Clone()`.

### Auth leans on env vars

For every provider, models.dev records the env-var names the provider expects to read credentials from (`env: ["OPENROUTER_API_KEY"]`). `sorus.New(ctx, cat, id)` does:

1. Looks up the provider record.
2. Reads its `env` list.
3. Takes the first name that is set in `os.LookupEnv`. Errors if none are.
4. Builds the internal client with that key + `provider.API` as the base URL.

We deliberately do not implement OAuth/device-code/browser flows for any provider. The `github-copilot` provider expects `GITHUB_TOKEN`, which is normally populated by `gh auth token` — that's a *user-side* concern. Setting that env var to a valid Copilot token is your job; we just read it.

Override the env-var lookup with `WithAPIKey(...)`:

```go
client, err := sorus.New(ctx, cat, "openrouter",
    sorus.WithAPIKey(os.Getenv("MY_APP_OPENROUTER_KEY")),
)
```

Or inject a custom `*http.Client` for retries/middleware/observability:

```go
client, err := sorus.New(ctx, cat, "openrouter",
    sorus.WithHTTPClient(myHTTPClient),
)
```

### Provider shapes supported today

`sorus.New(...)` accepts a provider when its `npm` field is one of:

- `@ai-sdk/openai-compatible` — generic chat-completions wire protocol with a custom base URL (181 providers, ~80% of the catalog)
- `@ai-sdk/openai` — the chat-completions shape with first-party body params (6: `openai`, `vivgrid`, `infer`, `meta`, `perplexity-agent`, `neosmith`)

Anything else (`anthropic`, `cohere`, `watsonx`, `bedrock-runtime`, `vertex`, `gemini`, …) returns `ErrUnsupportedProvider`. The deferred providers are queued behind dedicated implementations (see Roadmap below).

### Reasoning, multimodal, capabilities

The catalog records what each model can do (modalities, limits, cost, reasoning capability, tool-calling, etc.) and `Model` exposes those as plain fields. Configuration knobs you set on `Request` — `.Reasoning(...)`, `.ResponseFormat(...)`, `.Tools(...)` — apply at call time.

The catalog's per-model `ReasoningOptions` shape (one of three):

```go
type ReasoningOption struct {
    Type   ReasoningOptionType   // "effort" | "toggle" | "budget_tokens"
    Values []string              // when Type == "effort"
    Min    *int                  // when Type == "budget_tokens"
}
```

The generic `Reasoning` struct you set on `Request`:

```go
type Reasoning struct {
    Effort       string  // "low" | "medium" | "high" | "xhigh" | ...
    BudgetTokens int     // 0 = unset
}
```

Each implementation maps the generic `Reasoning` struct to its provider's idiom:

| `Reasoning` field | Chat completions (today) | Anthropic (planned) | Gemini (planned) | DeepSeek (planned) |
| --- | --- | --- | --- | --- |
| `Effort` | `reasoning_effort` | tier-mapped to budget | tier-mapped to budget range | `reasoning_effort` (low/med → high) |
| `BudgetTokens` | — | `thinking.budget_tokens` | `thinking_budget` | `thinking_budget` |

The chat-completions impl only honors `Effort`; the others are planned. Each implementation reads only the fields it understands; over-populating is harmless.

## Public API surface

### Catalog

```go
func LoadCatalog(ctx context.Context, opts ...CatalogOption) (*Catalog, error)
func LoadCatalogFromBytes(b []byte) (*Catalog, error)
func LoadCatalogFromFS(fsys fs.FS, name string) (*Catalog, error)

type CatalogOption func(*catalogConfig)
func WithRefresh() CatalogOption           // bypass the in-memory cache and re-fetch
func WithCatalogURL(string) CatalogOption  // override the default https://models.dev/api.json (also bypasses the cache; use a file:// URI for a vendored copy)
func WithHTTPClient(*http.Client) CatalogOption  // custom *http.Client for the fetch (also bypasses the cache)

// LoadCatalog caches the parsed catalog in memory for the lifetime of the
// process. Call WithRefresh() to force a re-fetch. LoadCatalogFromBytes
// and LoadCatalogFromFS never touch the network and bypass the cache.

type Catalog struct { /* unexported */ }

func (c *Catalog) Provider(id string) *Provider
func (c *Catalog) MustProvider(id string) *Provider
func (c *Catalog) Providers() []*Provider
func (c *Catalog) Models() []*Model
func (c *Catalog) ModelsByFamily(family string) []*Model
func (c *Catalog) FindModel(qualifiedID string) (*Provider, *Model, bool)
```

### Provider + Model (pure description)

```go
type Provider struct {
    ID     string    // "openrouter"
    NPM    string    // "@ai-sdk/openai-compatible"
    Name   string    // "OpenRouter"
    Env    []string  // ["OPENROUTER_API_KEY"]
    API    string    // "https://openrouter.ai/api/v1"
    Doc    string    // docs URL
    Models map[string]*Model
}

func (p *Provider) Model(id string) (*Model, bool)
func (p *Provider) MustModel(id string) *Model
func (p *Provider) Models() []*Model
func (p *Provider) ModelsByFamily(family string) []*Model

// Model is pure description. Nothing on it can be configured.
type Model struct {
    ID               string             // provider-local id, e.g. "anthropic/claude-sonnet-4-5"
    Name             string             // "Claude Sonnet 4.5"
    Family           string             // "claude-sonnet"
    Description      string
    Modalities       ModelModalities    // {input: ["text","image","pdf"], output: ["text"]}
    Limit            ModelLimit         // {context, input, output}
    Cost             ModelCost          // per-1M-token USD prices
    Reasoning        bool
    ReasoningOptions []ReasoningOption  // effort enum / toggle / budget_tokens
    ToolCall         bool
    StructuredOutput bool
    Temperature      bool
    Attachment       bool
    Knowledge        string             // "YYYY-MM"
    ReleaseDate      string
    LastUpdated      string
    Status           string             // "", "alpha", "beta", "deprecated"
    OpenWeights      bool
}

type ModelModalities struct{ Input, Output []string }
type ModelLimit        struct{ Context, Input, Output int }
type ModelCost         struct {
    Input, Output, CacheRead, CacheWrite, Reasoning float64
    InputAudio, OutputAudio                         float64
}
type ReasoningOptionType string
const (
    ReasoningEffort      ReasoningOptionType = "effort"
    ReasoningToggle      ReasoningOptionType = "toggle"
    ReasoningBudgetTokens ReasoningOptionType = "budget_tokens"
)
type ReasoningOption struct {
    Type   ReasoningOptionType
    Values []string
    Min    *int
}

func (m *Model) Ref() string         // == m.ID
func (m *Model) Provider() *Provider // back-reference to the parent provider
```

### Generic call surface

```go
// Construction (today only chat-completions; more to come).
func New(ctx context.Context, cat *Catalog, providerID string, opts ...Option) (Client, error)

type Option func(*config)
func WithAPIKey(string) Option
func WithHTTPClient(*http.Client) Option
func WithLogger(*slog.Logger) Option

// Client is the provider-agnostic surface.
type Client interface {
    Chat(ctx context.Context, req *Request) (*Response, error)
    Stream(ctx context.Context, req *Request) (*Stream, error)
}

// Request is built by fluent configuration. All setter methods take literal
// values and return *Request for chaining.
type Request struct { /* unexported fields */ }

func NewRequest(model *Model, messages ...Message) *Request

// Conversation:
func (r *Request) Message(msg Message) *Request  // append any Message
func (r *Request) System(text string) *Request   // append a plain system turn
func (r *Request) User(text string) *Request     // append a plain user turn
func (r *Request) Assistant(text string) *Request // append a plain assistant turn

// Tools:
func (r *Request) Tools(tools ...Tool) *Request
func (r *Request) ToolChoice(choice ToolChoice) *Request

// Model parameters:
func (r *Request) Temperature(v float64) *Request
func (r *Request) MaxTokens(v int) *Request
func (r *Request) TopP(v float64) *Request
func (r *Request) Stop(seqs ...string) *Request
func (r *Request) Reasoning(r Reasoning) *Request
func (r *Request) ResponseFormat(rf *ResponseFormat) *Request

// Inspection / reuse:
func (r *Request) Model() *Model
func (r *Request) Messages() []Message
func (r *Request) Tools() []Tool
func (r *Request) Clone() *Request

// Message: the unit of conversation.
type Message struct {
    Role Role
    Content []Part
}

type Role string
const (
    RoleSystem    Role = "system"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleTool      Role = "tool"
)

// Convenience constructors — sugar at the Message-constructor level so
// users who don't want fluent style can build []Message literals.
func SystemMessage(text string) Message
func UserMessage(text string) Message
func AssistantMessage(text string) Message

// Part is a sealed content-part interface — only Text, ImageURL,
// ImageData, and ToolResult are supported.
type Part interface{ isPart() }

type Text       struct{ Value string }
type ImageURL   struct{ URL, Detail string }
type ImageData  struct{ Data []byte; MIMEType string }
type ToolResult struct {
    ToolCallID string
    Content    []Part
    IsError    bool
}

// Tool: declared function for tool calling.
type Tool struct {
    Name        string
    Description string
    Parameters  map[string]any // JSON Schema as a Go map
}

type ToolChoice struct {
    Mode ToolChoiceMode
    Name string // required when Mode == ToolChoiceSpecific
}
type ToolChoiceMode string
const (
    ToolChoiceAuto     ToolChoiceMode = "auto"
    ToolChoiceNone     ToolChoiceMode = "none"
    ToolChoiceRequired ToolChoiceMode = "required"
    ToolChoiceSpecific ToolChoiceMode = "specific"
)

// Reasoning configures model reasoning.
type Reasoning struct {
    Effort       string  // "low" | "medium" | "high" | "xhigh" | ...
    BudgetTokens int     // 0 = unset
}

// ResponseFormat: structured output.
type ResponseFormat struct {
    Type       string // "json_object" | "json_schema"
    JSONSchema *JSONSchema
}
type JSONSchema struct {
    Name   string
    Schema map[string]any
    Strict bool
}

// Response: what you get back from Chat.
type Response struct {
    ID         string
    Model      string
    Message    Message
    StopReason string  // "end_turn" | "max_tokens" | "tool_use" | "stop_sequence"
    Usage      Usage
    Raw        any     // provider-specific raw payload; nil if the implementation doesn't expose one
}

func (m Message) Text() string  // concatenate Text parts; useful for plain-text responses

type Usage struct {
    InputTokens, OutputTokens, ReasoningTokens int
    CacheReadTokens, CacheWriteTokens, TotalTokens int
}

// Streaming surface.
type Stream struct{ /* ... */ }
func (s *Stream) Next() bool
func (s *Stream) Event() Event
func (s *Stream) Close() error
func (s *Stream) Err() error
func (s *Stream) Collect() (*Response, error)  // read to end and return the full Response

type EventType string
const (
    EventContentStart     EventType = "content_start"
    EventContentDelta     EventType = "content_delta"
    EventContentEnd       EventType = "content_end"
    EventReasoningStart   EventType = "reasoning_start"
    EventReasoningDelta   EventType = "reasoning_delta"
    EventReasoningEnd     EventType = "reasoning_end"
    EventToolCallStart    EventType = "tool_call_start"
    EventToolCallDelta    EventType = "tool_call_delta"
    EventToolCallEnd      EventType = "tool_call_end"
    EventDone             EventType = "done"
    EventError            EventType = "error"
)

type Event struct {
    Type           EventType
    TextDelta      string
    ReasoningDelta string
    ToolCall       *ToolCall
    ArgumentDelta  string
    StopReason     string
    Usage          *Usage
    Message        *Message
    Err            error
}

// ToolCall: emitted by the model in a Response or stream.
type ToolCall struct {
    ID        string
    Name      string
    Arguments string  // raw JSON; json.Unmarshal to get structured input
}
```

### Errors

```
ErrUnknownProvider       // provider id not in catalog
ErrUnsupportedProvider   // provider's npm is not @ai-sdk/openai-compatible or @ai-sdk/openai
ErrMissingAPIKey         // no env var from provider.Env is set, and no WithAPIKey override
ErrCatalogFetch         // network/HTTP error fetching api.json
ErrCatalogParse         // JSON parse / schema violation
ErrStreamClosed          // operations on a closed Stream
```

## Recipes

### Stream a response

```go
req := sorus.NewRequest(model).
    System("You are a helpful assistant.").
    Temperature(0.7).
    MaxTokens(2048).
    User("Tell me a story.")

stream, err := client.Stream(ctx, req)
if err != nil { panic(err) }
defer stream.Close()

var out strings.Builder
for stream.Next() {
    ev := stream.Event()
    switch ev.Type {
    case sorus.EventContentDelta:
        out.WriteString(ev.TextDelta)
        fmt.Print(ev.TextDelta)
    case sorus.EventReasoningDelta:
        // collect or display; e.g. ui.Spin(ev.ReasoningDelta)
    case sorus.EventDone:
        fmt.Printf("\n--- done, usage=%+v ---\n", ev.Usage)
    case sorus.EventError:
        return ev.Err
    }
}

// Or skip the loop entirely:
resp, err := stream.Collect()
```

### Tool calling

```go
type GetWeatherArgs struct {
    Location string `json:"location"`
}

req := sorus.NewRequest(model).
    System("...").
    User("Weather in Paris?").
    Tools(sorus.Tool{
        Name:        "get_weather",
        Description: "Get the current weather for a location.",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "location": map[string]any{"type": "string"},
            },
            "required": []string{"location"},
        },
    }).
    ToolChoice(sorus.ToolChoice{Mode: sorus.ToolChoiceRequired, Name: "get_weather"})

resp, err := client.Chat(ctx, req)

// If the model called a tool, run it and reply
if len(resp.Message.ToolCalls) > 0 {
    var toolResults []sorus.Part
    for _, tc := range resp.Message.ToolCalls {
        var args GetWeatherArgs
        _ = json.Unmarshal([]byte(tc.Arguments), &args)
        toolResults = append(toolResults, sorus.ToolResult{
            ToolCallID: tc.ID,
            Content: []sorus.Part{
                sorus.Text{Value: fetchWeather(ctx, args.Location)},
            },
        })
    }

    // Append the assistant turn + tool result turn, send again
    req = req.Clone().
        Message(resp.Message).                          // the assistant's tool-call turn
        Message(sorus.Message{
            Role:    sorus.RoleTool,
            Content: toolResults,
        })

    resp, err = client.Chat(ctx, req)
}
```

### Structured output (JSON schema)

```go
req := sorus.NewRequest(model).
    User("Summarize today's weather in Paris.").
    ResponseFormat(&sorus.ResponseFormat{
        Type: "json_schema",
        JSONSchema: &sorus.JSONSchema{
            Name: "weather_report",
            Strict: true,
            Schema: map[string]any{
                "type": "object",
                "properties": map[string]any{
                    "summary": map[string]any{"type": "string"},
                    "temp_f":  map[string]any{"type": "number"},
                },
                "required": []string{"summary", "temp_f"},
            },
        },
    })

resp, err := client.Chat(ctx, req)
```

### Native reasoning effort

```go
req := sorus.NewRequest(model).
    User("Solve this step-by-step.").
    Reasoning(sorus.Reasoning{Effort: "medium"})

resp, err := client.Chat(ctx, req)
```

### Multimodal input

```go
req := sorus.NewRequest(model).
    Message(sorus.Message{
        Role: sorus.RoleUser,
        Content: []sorus.Part{
            sorus.Text{Value: "What's in this image?"},
            sorus.ImageURL{URL: "https://example.com/cat.jpg", Detail: "high"},
            // Or:
            sorus.ImageData{Data: pngBytes, MIMEType: "image/png"},
        },
    })

resp, err := client.Chat(ctx, req)
```

### Find all Claude Sonnet models across providers

```go
sonnets := cat.ModelsByFamily("claude-sonnet")
for _, m := range sonnets {
    fmt.Printf("%s (%s): $%.2f/Mtok in\n", m.Ref(), m.Provider().ID, m.Cost.Input)
}
```

### Vendored / offline usage

```go
//go:embed api.json
var apiJSON []byte

func loadCatalog(ctx context.Context) (*sorus.Catalog, error) {
    return sorus.LoadCatalogFromBytes(apiJSON)
}
```

### Reusable defaults with `Clone`

```go
defaultReq := sorus.NewRequest(model).
    System("You are a helpful assistant.").
    Temperature(0.7).
    MaxTokens(2048)

for _, query := range queries {
    req := defaultReq.Clone().User(query)
    resp, err := client.Chat(ctx, req)
    if err != nil { /* ... */ }
}
```

`Clone()` produces an independent `*Request` you can extend per call without touching `defaultReq`.

## Caveats

- **No OAuth.** Setting `GITHUB_TOKEN` to a valid Copilot token via `gh auth` is fine; running an OAuth browser flow inside this package is not.
- **No protocol-specific escape hatch.** If the `Client` interface doesn't expose something you need, file an issue. We deliberately don't expose wire-protocol SDK types from the package's public surface.
- **`Tool.Parameters` is raw `map[string]any`.** Build JSON Schema manually (or with `invopop/jsonschema`); we don't ship a builder.
- **`Stream.Event()` returns a value type, not a pointer.** The hot path allocates an `Event` per token. If that's a problem for your workload, we can add a `Stream.Visit(func(Event) bool)` variant. Today's default is the simpler shape.
- **Catalog freshness.** We rely on `https://models.dev/api.json` and cache the parsed catalog in memory for the process lifetime. For reproducible builds, prefer vendoring via `LoadCatalogFromBytes` or `LoadCatalogFromFS` so the network isn't touched.
- **First-party TOML not consumed.** The upstream repo also publishes detailed per-provider TOML with merged lab metadata. We're consuming the simpler `api.json` view.
- **No provider-specific request shaping beyond auth + base URL.** Hosts with extra conventions (OpenRouter's `HTTP-Referer`/`X-Title`, DeepSeek's `thinking.type`, Bedrock's `converse` vs `invokeModel`) are routed through the generic `Request` shape. Provider-specific extensions can be added per implementation via separate interfaces or `Options`, but we don't ship them in v1.

## Roadmap

In rough order of value:

- **Anthropic native implementation** — covers `anthropic`, `subconscious`, `freemodel`, `minimax`, `minimax-cn`, `thinkingmachines`, and `google-vertex-anthropic`.
- **Amazon Bedrock Runtime** — covers `amazon-bedrock`.
- **Vertex AI + Gemini Developer API** — covers `google-vertex`, `google`.
- **Watsonx** — covers `watsonx`.
- **Per-provider reasoning mapping** — translate `Reasoning` into each implementation's native params (e.g. Anthropic `thinking.{type, budget_tokens}`).
- **Embeddings** — single new method on `Client`.
- **TOML loader** — consume upstream `sst/models.dev` TOML repo directly for richer lab metadata.

## License

**MIT.** Place the standard MIT license text in a `LICENSE` file at the repo root. SPDX identifier: `MIT`. Copyright line: `Copyright (c) 2026 Alan Johnson`.
