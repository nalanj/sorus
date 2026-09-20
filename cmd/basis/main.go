// Command basis is a small CLI for talking to any models.dev provider that
// basis supports. It loads the catalog, builds a Client for the requested
// provider (resolving the API key from the provider's documented env vars),
// and either streams the response or returns it as a single shot.
//
// Usage:
//
//	MINIMAX_API_KEY=... basis -prompt "Hello!"
//	MINIMAX_API_KEY=... basis -provider openrouter -model anthropic/claude-sonnet-4-5 -prompt "Hello!"
//	echo "Hi there" | MINIMAX_API_KEY=... basis
//
// Run `basis -h` for flags.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tokagen/basis"
)

func main() {
	provider := flag.String("provider", "minimax",
		"provider id in the models.dev catalog (e.g. \"openrouter\", \"anthropic\", \"minimax\")")
	modelID := flag.String("model", "MiniMax-M2",
		"model id on the chosen provider (e.g. \"MiniMax-M2\", \"anthropic/claude-sonnet-4-5\")")
	system := flag.String("system", "",
		"system prompt (sent as a system message)")
	promptFlag := flag.String("prompt", "",
		"user prompt; if empty, read from stdin")
	temperature := flag.Float64("temperature", -1,
		"sampling temperature in 0.0–1.0; -1 leaves it unset")
	maxTokens := flag.Int("max-tokens", -1,
		"maximum number of output tokens; -1 leaves it unset")
	noStream := flag.Bool("no-stream", false,
		"make a single non-streaming request instead of streaming")
	quiet := flag.Bool("quiet", false,
		"suppress non-text events (reasoning deltas, done summary)")
	flag.Parse()

	if err := run(context.Background(), runOpts{
		provider:    *provider,
		modelID:     *modelID,
		system:      *system,
		promptFlag:  *promptFlag,
		temperature: *temperature,
		maxTokens:   *maxTokens,
		noStream:    *noStream,
		quiet:       *quiet,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "basis: %v\n", err)
		os.Exit(1)
	}
}

type runOpts struct {
	provider    string
	modelID     string
	system      string
	promptFlag  string
	temperature float64
	maxTokens   int
	noStream    bool
	quiet       bool
}

func run(ctx context.Context, o runOpts) error {
	cat, err := basis.LoadCatalog(ctx)
	if err != nil {
		return fmt.Errorf("load catalog: %w", err)
	}

	client, err := basis.New(ctx, cat, o.provider)
	if err != nil {
		return fmt.Errorf("client: %w", err)
	}

	prov := cat.Provider(o.provider)
	if prov == nil {
		return fmt.Errorf("provider %q not in catalog", o.provider)
	}
	m, ok := prov.Model(o.modelID)
	if !ok {
		return fmt.Errorf("model %q not found on provider %q", o.modelID, o.provider)
	}

	req := basis.NewRequest(m)
	if o.system != "" {
		req = req.System(o.system)
	}
	if o.temperature >= 0 {
		req = req.Temperature(o.temperature)
	}
	if o.maxTokens > 0 {
		req = req.MaxTokens(o.maxTokens)
	}

	prompt := strings.TrimSpace(o.promptFlag)
	if prompt == "" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
		prompt = strings.TrimSpace(string(data))
	}
	if prompt == "" {
		return fmt.Errorf("no prompt provided (use -prompt or pipe via stdin)")
	}
	req = req.User(prompt)

	if o.noStream {
		resp, err := client.Chat(ctx, req)
		if err != nil {
			return fmt.Errorf("chat: %w", err)
		}
		out := resp.Message.Text()
		if out == "" {
			return fmt.Errorf("no content in response (stop_reason=%s)", resp.StopReason)
		}
		fmt.Println(out)
		return nil
	}

	stream, err := client.Stream(ctx, req)
	if err != nil {
		return fmt.Errorf("stream: %w", err)
	}
	defer stream.Close()

	for stream.Next() {
		ev := stream.Event()
		switch ev.Type {
		case basis.EventContentDelta:
			fmt.Print(ev.TextDelta)
		case basis.EventContentStart:
			// marker event; ignore
		case basis.EventReasoningDelta:
			if !o.quiet {
				fmt.Fprintf(os.Stderr, "[reasoning] %s", ev.ReasoningDelta)
			}
		case basis.EventToolCallStart:
			if !o.quiet {
				fmt.Fprintf(os.Stderr, "[tool_call] %s(%s)\n", ev.ToolCall.Name, ev.ToolCall.ID)
			}
		case basis.EventToolCallDelta:
			if !o.quiet {
				fmt.Fprintf(os.Stderr, "  args += %s\n", ev.ArgumentDelta)
			}
		case basis.EventToolCallEnd:
			// marker event; ignore
		case basis.EventDone:
			if !o.quiet {
				if ev.Usage != nil {
					fmt.Fprintf(os.Stderr, "\n[done: stop=%s in_tok=%d out_tok=%d]\n",
						ev.StopReason, ev.Usage.InputTokens, ev.Usage.OutputTokens)
				} else {
					fmt.Fprintf(os.Stderr, "\n[done: stop=%s]\n", ev.StopReason)
				}
			}
		case basis.EventError:
			return fmt.Errorf("stream: %w", ev.Err)
		}
	}
	if err := stream.Err(); err != nil {
		return fmt.Errorf("stream end: %w", err)
	}
	fmt.Println()
	return nil
}
