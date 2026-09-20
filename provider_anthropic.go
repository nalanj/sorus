package sorus

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicClient implements [Client] for any provider whose npm is
// `@ai-sdk/anthropic` (Anthropic Messages API). Supports custom base URLs
// (the `Provider.API` field) so any Anthropic-compatible gateway also works.
type anthropicClient struct {
	api      anthropic.Client
	provider *Provider
}

// anthropicBaseURL trims a trailing "/v1" from the catalog's API URL so we
// don't duplicate the version segment; the Anthropic Go SDK appends
// "/v1/messages" to whatever BaseURL is supplied. Catalog URLs were
// generated for the AI SDK ecosystem (where the trailing "/v1" stays).
func anthropicBaseURL(api string) string {
	if api == "" {
		return ""
	}
	api = strings.TrimRight(api, "/")
	api = strings.TrimSuffix(api, "/v1")
	return api
}

func newAnthropicClient(p *Provider, apiKey string, hc *http.Client) (Client, error) {
	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if base := anthropicBaseURL(p.API); base != "" {
		opts = append(opts, option.WithBaseURL(base))
	}
	if hc != nil {
		opts = append(opts, option.WithHTTPClient(hc))
	}
	api := anthropic.NewClient(opts...)
	return &anthropicClient{api: api, provider: p}, nil
}

func (c *anthropicClient) Chat(ctx context.Context, req *Request) (*Response, error) {
	params, err := buildAnthropicParams(req)
	if err != nil {
		return nil, err
	}
	raw, err := c.api.Messages.New(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertAnthropicMessage(raw)
}

func (c *anthropicClient) Stream(ctx context.Context, req *Request) (*Stream, error) {
	params, err := buildAnthropicParams(req)
	if err != nil {
		return nil, err
	}
	stream := c.api.Messages.NewStreaming(ctx, params)
	return wrapStream(func() (Event, bool, error) {
		if !stream.Next() {
			if err := stream.Err(); err != nil {
				return errorEvent(err), false, err
			}
			return Event{}, false, nil
		}
		ev, ok, err := convertAnthropicStreamEvent(stream.Current())
		if err != nil {
			return errorEvent(err), false, err
		}
		if !ok {
			return Event{}, true, nil
		}
		return ev, true, nil
	}), nil
}

// --- request building ---

// defaultAnthropicMaxTokens is used when the caller didn't set max_tokens and
// the chosen provider doesn't have a known limit. Anthropic requires
// `max_tokens` to be present.
const defaultAnthropicMaxTokens int64 = 4096

func buildAnthropicParams(req *Request) (anthropic.MessageNewParams, error) {
	p := anthropic.MessageNewParams{
		Model:    req.Model().Ref(),
		Messages: []anthropic.MessageParam{},
	}

	// Anthropic takes system messages at the top level (not in the
	// messages slice) and as text blocks. Pull any system-role messages
	// out and concatenate their content.
	var systemBlocks []anthropic.TextBlockParam
	var conversation []Message
	for _, m := range req.Messages() {
		if m.Role == RoleSystem {
			for _, p := range m.Content {
				if t, ok := p.(Text); ok {
					systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: t.Value})
				}
			}
			continue
		}
		conversation = append(conversation, m)
	}
	if len(systemBlocks) > 0 {
		p.System = systemBlocks
	}

	for _, m := range conversation {
		mp, err := buildAnthropicMessage(m)
		if err != nil {
			return p, fmt.Errorf("request: %w", err)
		}
		p.Messages = append(p.Messages, mp)
	}

	if tools := req.tools; len(tools) > 0 {
		p.Tools = buildAnthropicTools(tools)
	}
	if v := req.toolChoicePtr(); v != nil {
		p.ToolChoice = buildAnthropicToolChoice(*v)
	}
	if v := req.temperaturePtr(); v != nil {
		p.Temperature = anthropic.Float(*v)
	}
	if v := req.topPPtr(); v != nil {
		p.TopP = anthropic.Float(*v)
	}
	if v := req.maxTokensPtr(); v != nil {
		p.MaxTokens = int64(*v)
	} else {
		p.MaxTokens = defaultAnthropicMaxTokens
	}
	if seqs := req.stopSeqs(); len(seqs) > 0 {
		p.StopSequences = append([]string(nil), seqs...)
	}
	if r := req.reasoningPtr(); r != nil && r.BudgetTokens > 0 {
		p.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(r.BudgetTokens))
	}
	return p, nil
}

func buildAnthropicMessage(m Message) (anthropic.MessageParam, error) {
	switch m.Role {
	case RoleUser:
		blocks, err := buildAnthropicUserBlocks(m.Content)
		if err != nil {
			return anthropic.MessageParam{}, err
		}
		return anthropic.NewUserMessage(blocks...), nil
	case RoleAssistant:
		// Reconstruct: text content as TextBlock, tool calls as ToolUseBlock.
		var blocks []anthropic.ContentBlockParamUnion
		for _, p := range m.Content {
			if t, ok := p.(Text); ok {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{Text: t.Value},
				})
			}
		}
		for _, tc := range m.ToolCalls {
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					ID:    tc.ID,
					Name:  tc.Name,
					Input: json.RawMessage(tc.Arguments),
				},
			})
		}
		if len(blocks) == 0 {
			// Anthropic requires content. Use empty text as fallback.
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfText: &anthropic.TextBlockParam{Text: ""},
			})
		}
		return anthropic.MessageParam{
			Role:    "assistant",
			Content: blocks,
		}, nil
	case RoleTool:
		// Anthropic takes tool results as user-role messages with tool_result
		// blocks. If multiple results, they're all under one user turn.
		var blocks []anthropic.ContentBlockParamUnion
		for _, p := range m.Content {
			tr, ok := p.(ToolResult)
			if !ok {
				continue
			}
			var inner []anthropic.ToolResultBlockParamContentUnion
			for _, sp := range tr.Content {
				if t, ok := sp.(Text); ok {
					inner = append(inner, anthropic.ToolResultBlockParamContentUnion{
						OfText: &anthropic.TextBlockParam{Text: t.Value},
					})
				}
			}
			r := &anthropic.ToolResultBlockParam{
				ToolUseID: tr.ToolCallID,
				Content:   inner,
				IsError:   anthropic.Bool(tr.IsError),
			}
			blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolResult: r})
		}
		return anthropic.NewUserMessage(blocks...), nil
	default:
		return anthropic.MessageParam{}, fmt.Errorf("anthropic: unsupported role %q", m.Role)
	}
}

func buildAnthropicUserBlocks(parts []Part) ([]anthropic.ContentBlockParamUnion, error) {
	if len(parts) == 1 {
		if t, ok := parts[0].(Text); ok {
			return []anthropic.ContentBlockParamUnion{
				{OfText: &anthropic.TextBlockParam{Text: t.Value}},
			}, nil
		}
	}
	var out []anthropic.ContentBlockParamUnion
	for _, p := range parts {
		switch pp := p.(type) {
		case Text:
			out = append(out, anthropic.ContentBlockParamUnion{
				OfText: &anthropic.TextBlockParam{Text: pp.Value},
			})
		case ImageURL:
			out = append(out, anthropic.ContentBlockParamUnion{
				OfImage: &anthropic.ImageBlockParam{
					Source: anthropic.ImageBlockParamSourceUnion{
						OfURL: &anthropic.URLImageSourceParam{URL: pp.URL},
					},
				},
			})
		case ImageData:
			mt, err := anthropicMediaType(pp.MIMEType)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropic.ContentBlockParamUnion{
				OfImage: &anthropic.ImageBlockParam{
					Source: anthropic.ImageBlockParamSourceUnion{
						OfBase64: &anthropic.Base64ImageSourceParam{
							Data:      base64.StdEncoding.EncodeToString(pp.Data),
							MediaType: mt,
						},
					},
				},
			})
		default:
			// tool results / unsupported: skip (caller is expected to put
			// tool results in RoleTool messages, not user ones).
		}
	}
	return out, nil
}

// anthropicMediaType maps an incoming MIME type to one Anthropic accepts on
// the Base64ImageSource. Returns an error for unsupported types.
func anthropicMediaType(mt string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mt {
	case "image/jpeg", "image/jpg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	}
	return "", fmt.Errorf("anthropic: unsupported image MIME type %q", mt)
}

func buildAnthropicTools(tools []Tool) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{
			Type: "object",
		}
		if props, ok := t.Parameters["properties"]; ok {
			schema.Properties = props
		}
		if req, ok := t.Parameters["required"]; ok {
			if arr, ok := req.([]any); ok {
				for _, v := range arr {
					if s, ok := v.(string); ok {
						schema.Required = append(schema.Required, s)
					}
				}
			} else if arr, ok := req.([]string); ok {
				schema.Required = append(schema.Required, arr...)
			}
		}
		out = append(out, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: schema,
			},
		})
	}
	return out
}

func buildAnthropicToolChoice(tc ToolChoice) anthropic.ToolChoiceUnionParam {
	switch tc.Mode {
	case ToolChoiceNone:
		n := anthropic.NewToolChoiceNoneParam()
		return anthropic.ToolChoiceUnionParam{OfNone: &n}
	case ToolChoiceAuto:
		v := anthropic.ToolChoiceAutoParam{}
		return anthropic.ToolChoiceUnionParam{OfAuto: &v}
	case ToolChoiceRequired:
		v := anthropic.ToolChoiceAnyParam{}
		return anthropic.ToolChoiceUnionParam{OfAny: &v}
	case ToolChoiceSpecific:
		v := anthropic.ToolChoiceToolParam{Name: tc.Name}
		return anthropic.ToolChoiceUnionParam{OfTool: &v}
	}
	v := anthropic.ToolChoiceAutoParam{}
	return anthropic.ToolChoiceUnionParam{OfAuto: &v}
}

// --- response conversion (non-streaming) ---

func convertAnthropicMessage(raw *anthropic.Message) (*Response, error) {
	if raw == nil {
		return nil, fmt.Errorf("nil anthropic message")
	}
	msg := Message{Role: RoleAssistant}
	for _, block := range raw.Content {
		tb := block.AsText()
		if tb.Text != "" || block.Type == "text" {
			msg.Content = append(msg.Content, Text{Value: tb.Text})
			continue
		}
		if tub := block.AsToolUse(); tub.Name != "" || block.Type == "tool_use" {
			args := string(tub.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:        tub.ID,
				Name:      tub.Name,
				Arguments: args,
			})
		}
	}
	return &Response{
		ID:         raw.ID,
		Model:      string(raw.Model),
		Message:    msg,
		StopReason: string(raw.StopReason),
		Usage: Usage{
			InputTokens:      int(raw.Usage.InputTokens),
			OutputTokens:     int(raw.Usage.OutputTokens),
			CacheReadTokens:  int(raw.Usage.CacheReadInputTokens),
			CacheWriteTokens: int(raw.Usage.CacheCreationInputTokens),
		},
		Raw: raw,
	}, nil
}

// --- response conversion (streaming) ---

func convertAnthropicStreamEvent(ev anthropic.MessageStreamEventUnion) (Event, bool, error) {
	switch ev.Type {
	case "message_start":
		// Open the stream. The Message field carries the ID/Model. We
		// emit a content_start as a signal that an assistant message has
		// begun, with no payload.
		return Event{Type: EventContentStart}, true, nil
	case "content_block_start":
		// A new content block (text or tool_use) begins. For tool_use,
		// emit a tool_call_start so callers can allocate.
		cb := ev.ContentBlock.AsToolUse()
		if cb.ID != "" || ev.ContentBlock.Type == "tool_use" {
			return Event{
				Type: EventToolCallStart,
				ToolCall: &ToolCall{
					ID:   cb.ID,
					Name: cb.Name,
				},
			}, true, nil
		}
		return Event{Type: EventContentStart}, true, nil
	case "content_block_delta":
		delta := ev.Delta
		switch delta.Type {
		case "text_delta":
			return Event{Type: EventContentDelta, TextDelta: delta.Text}, true, nil
		case "thinking_delta":
			return Event{Type: EventReasoningDelta, ReasoningDelta: delta.Thinking}, true, nil
		case "input_json_delta":
			return Event{Type: EventToolCallDelta, ArgumentDelta: delta.PartialJSON}, true, nil
		}
		return Event{}, false, nil
	case "content_block_stop":
		// End of a tool_use block. The caller has accumulated the args via
		// input_json_delta; we don't have anything else to emit here.
		return Event{}, false, nil
	case "message_delta":
		// Stream-end metadata update (stop_reason, final usage).
		u := ev.Usage
		return Event{
			Type:       EventDone,
			StopReason: string(ev.Delta.StopReason),
			Usage: &Usage{
				InputTokens:      int(u.InputTokens),
				OutputTokens:     int(u.OutputTokens),
				CacheReadTokens:  int(u.CacheReadInputTokens),
				CacheWriteTokens: int(u.CacheCreationInputTokens),
			},
		}, true, nil
	case "message_stop":
		// Final shutdown event. If message_delta already emitted Done,
		// swallow this one.
		return Event{}, false, nil
	case "ping", "compact_message", "error":
		return Event{}, false, nil
	}
	return Event{}, false, nil
}
