# sorus

> A Go client for (mostly) any provider in the [models.dev](https://models.dev) catalog.

Sorus is meant primarily for cases where dynamic model usage is ideal.

Loads the catalog of providers and models into typed Go structs and gives you one `Client` to call any provider that speaks a supported wire protocol. Build a `Request` fluently, get a `Response` back.

Supported wire protocols: OpenAI chat completions (`@ai-sdk/openai-compatible`, `@ai-sdk/openai`) and Anthropic (`@ai-sdk/anthropic`). Anything else returns `ErrUnsupportedProvider`.

## Install

```bash
go get github.com/nalanj/sorus
```

Go 1.22+.

```go
import "github.com/nalanj/sorus"
```

## Quick start

```go
ctx := context.Background()

cat, err := sorus.LoadCatalog(ctx)
if err != nil {
    panic(err)
}

client, err := sorus.New(ctx, cat, "openrouter")
if err != nil {
    panic(err)
}

model := cat.Provider("openrouter").MustModel("anthropic/claude-sonnet-4-5")

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
```

`LoadCatalog` fetches `https://models.dev/api.json` and caches the parsed result for the process lifetime. `New(ctx, cat, "openrouter")` reads the env var the catalog says OpenRouter expects (`OPENROUTER_API_KEY`) and points at the right base URL. Set the key, call `Chat`, get a response.

## Design

### Wrapped, not pass-through

"OpenAI-compatible" describes the wire, not the API. Different providers expose different params — `reasoning_effort` here, `enable_thinking` there. `sorus` hides those differences behind one type. You import only `github.com/nalanj/sorus`; the package internally implements `Client` for each supported wire protocol and converts your fluent `Request` into the right shape.

There is no protocol-specific escape hatch by design. If `Client` misses something you need, that's the seam — open an issue.

### `Model` is description, `Request` is configuration

`Model` is read-only: name, family, modalities, limits, cost, capabilities, release date. Nothing on it can be configured.

`Request` is built fluently. All setters take literal values and return `*Request` for chaining. Reuse defaults with `req.Clone()`.

### Auth via env vars

The catalog records the env-var names each provider expects (`env: ["OPENROUTER_API_KEY"]`). `New` reads the first one that's set, errors if none are. OAuth flows aren't built in — populating `GITHUB_TOKEN` via `gh auth` for the `github-copilot` provider is your job.

Override per-call:

```go
client, err := sorus.New(ctx, cat, "openrouter",
    sorus.WithAPIKey(os.Getenv("MY_KEY")),
    sorus.WithHTTPClient(myHTTPClient),
)
```

### Reasoning, multimodal, capabilities

The catalog records what each model can do — modalities, limits, cost, reasoning capability, tool calling, structured output — and `Model` exposes those as plain fields. Configuration knobs go on `Request`.

The generic `Reasoning` struct maps onto each implementation's idiom:

| `Reasoning` field | Chat completions | Anthropic |
| --- | --- | --- |
| `Effort` | `reasoning_effort` | — |
| `BudgetTokens` | — | `thinking.budget_tokens` |

Each implementation reads only the fields it understands; over-populating is harmless.

## API reference

Full type and function reference lives on [pkg.go.dev/github.com/nalanj/sorus](https://pkg.go.dev/github.com/nalanj/sorus).

## Recipes

### Stream a response

```go
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
            Name:   "weather_report",
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
```

### Reasoning

```go
// Chat-completions providers honor Effort:
req := sorus.NewRequest(model).
    User("Solve this step-by-step.").
    Reasoning(sorus.Reasoning{Effort: "medium"})

// Anthropic honors BudgetTokens:
req = sorus.NewRequest(model).
    User("Solve this step-by-step.").
    Reasoning(sorus.Reasoning{BudgetTokens: 4096})
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

func loadCatalog() (*sorus.Catalog, error) {
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

- **No OAuth flows.** Setting `GITHUB_TOKEN` to a valid Copilot token via `gh auth` is fine; running an OAuth browser flow inside this package is not.
- **No protocol-specific escape hatch.** If `Client` doesn't expose what you need, that's the seam — open an issue. The package deliberately doesn't expose wire-protocol SDK types from its public surface.
- **`Tool.Parameters` is raw `map[string]any`.** Build JSON Schema manually (or with `invopop/jsonschema`); we don't ship a builder.
- **`Stream.Event()` returns a value type, not a pointer.** The hot path allocates an `Event` per token. If that's a problem for your workload, a `Stream.Visit(func(Event) bool)` variant would be straightforward to add.
- **Catalog freshness.** `LoadCatalog` relies on `https://models.dev/api.json` and caches the parsed catalog in memory for the process lifetime. For reproducible builds, prefer vendoring via `LoadCatalogFromBytes` or `LoadCatalogFromFS` so the network isn't touched.

## License

**MIT.** See `LICENSE` for details.
