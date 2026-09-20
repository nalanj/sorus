package basis

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// chatCompletionsClient implements [Client] for any provider whose npm is
// `@ai-sdk/openai-compatible` or `@ai-sdk/openai`. The wire is OpenAI's
// chat-completions protocol; we translate the generic [Request] into
// openai-go params, send, and translate the response back.
type chatCompletionsClient struct {
	api      openai.Client
	provider *Provider
}

func newChatCompletionsClient(p *Provider, apiKey string, hc *http.Client) (Client, error) {
	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if p.API != "" {
		opts = append(opts, option.WithBaseURL(p.API))
	}
	if hc != nil {
		opts = append(opts, option.WithHTTPClient(hc))
	}
	api := openai.NewClient(opts...)
	return &chatCompletionsClient{api: api, provider: p}, nil
}

func (c *chatCompletionsClient) Chat(ctx context.Context, req *Request) (*Response, error) {
	params, err := buildChatParams(req)
	if err != nil {
		return nil, err
	}
	raw, err := c.api.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertChatCompletion(raw)
}

func (c *chatCompletionsClient) Stream(ctx context.Context, req *Request) (*Stream, error) {
	params, err := buildChatParams(req)
	if err != nil {
		return nil, err
	}
	stream := c.api.Chat.Completions.NewStreaming(ctx, params)
	if stream == nil {
		return nil, fmt.Errorf("basis: failed to open stream")
	}
	return wrapStream(func() (Event, bool, error) {
		if !stream.Next() {
			if err := stream.Err(); err != nil {
				return errorEvent(err), false, err
			}
			return Event{}, false, nil
		}
		ev, hasContent, err := convertChatChunk(stream.Current())
		if err != nil {
			return errorEvent(err), false, err
		}
		if !hasContent {
			// skip chunk
			return Event{}, true, nil
		}
		return ev, true, nil
	}), nil
}

// --- request building ---

func buildChatParams(req *Request) (openai.ChatCompletionNewParams, error) {
	p := openai.ChatCompletionNewParams{
		Model: req.Model().Ref(),
	}

	msgs, err := buildOpenAIMessages(req.Messages())
	if err != nil {
		return p, fmt.Errorf("request: %w", err)
	}
	p.Messages = msgs

	if tools := req.tools; len(tools) > 0 {
		p.Tools = buildOpenAITools(tools)
	}
	if v := req.toolChoicePtr(); v != nil {
		oc := buildOpenAIToolChoice(*v)
		p.ToolChoice = oc
	}
	if v := req.temperaturePtr(); v != nil {
		p.Temperature = openai.Float(*v)
	}
	if v := req.maxTokensPtr(); v != nil {
		p.MaxTokens = openai.Int(int64(*v))
	}
	if v := req.topPPtr(); v != nil {
		p.TopP = openai.Float(*v)
	}
	if seqs := req.stopSeqs(); len(seqs) > 0 {
		if len(seqs) == 1 {
			p.Stop = openai.ChatCompletionNewParamsStopUnion{OfString: openai.String(seqs[0])}
		} else {
			arr := append([]string(nil), seqs...)
			p.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: arr}
		}
	}
	if r := req.reasoningPtr(); r != nil && r.Effort != "" {
		p.ReasoningEffort = shared.ReasoningEffort(r.Effort)
	}
	if rf := req.responseFormatPtr(); rf != nil {
		p.ResponseFormat = buildOpenAIResponseFormat(rf)
	}
	return p, nil
}

func buildOpenAIMessages(msgs []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for i, m := range msgs {
		u, err := buildOpenAIMessage(m)
		if err != nil {
			return nil, fmt.Errorf("message %d (%s): %w", i, m.Role, err)
		}
		out = append(out, u)
	}
	return out, nil
}

func buildOpenAIMessage(m Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch m.Role {
	case RoleSystem:
		return openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: textContentOf(m.Content),
				Role:    "system",
			},
		}, nil
	case RoleUser:
		return openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: contentUnionOf(m.Content),
				Role:    "user",
			},
		}, nil
	case RoleAssistant:
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Role:    "assistant",
				Content: assistantTextContentOf(m.Content),
			},
		}, nil
	case RoleTool:
		out := openai.ChatCompletionMessageParamUnion{}
		tool := &openai.ChatCompletionToolMessageParam{
			Role: "tool",
		}
		// openai-go's tool message takes a single string Content.
		tool.Content = openai.ChatCompletionToolMessageParamContentUnion{
			OfString: openai.String(joinedText(flattenToolResults(m.Content))),
		}
		// The first ToolResult provides ToolCallID; if multiple, the IDs
		// in subsequent parts are lost. We pick the first and warn? v1:
		// pick first, ignore the rest.
		for _, p := range m.Content {
			if tr, ok := p.(ToolResult); ok {
				tool.ToolCallID = tr.ToolCallID
				break
			}
		}
		out.OfTool = tool
		return out, nil
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown role %q", m.Role)
	}
}

// textContentOf returns a system-message content union (text only).
func textContentOf(parts []Part) openai.ChatCompletionSystemMessageParamContentUnion {
	return openai.ChatCompletionSystemMessageParamContentUnion{
		OfString: openai.String(joinedText(parts)),
	}
}

// assistantTextContentOf returns an assistant-message content union (text only).
func assistantTextContentOf(parts []Part) openai.ChatCompletionAssistantMessageParamContentUnion {
	return openai.ChatCompletionAssistantMessageParamContentUnion{
		OfString: openai.String(joinedText(parts)),
	}
}

func contentUnionOf(parts []Part) openai.ChatCompletionUserMessageParamContentUnion {
	if len(parts) == 1 {
		if t, ok := parts[0].(Text); ok {
			return openai.ChatCompletionUserMessageParamContentUnion{OfString: openai.String(t.Value)}
		}
	}
	arr := make([]openai.ChatCompletionContentPartUnionParam, 0, len(parts))
	for _, p := range parts {
		switch pp := p.(type) {
		case Text:
			arr = append(arr, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: pp.Value},
			})
		case ImageURL:
			arr = append(arr, openai.ChatCompletionContentPartUnionParam{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: pp.URL, Detail: pp.Detail},
				},
			})
		case ImageData:
			dataURL := "data:" + pp.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(pp.Data)
			arr = append(arr, openai.ChatCompletionContentPartUnionParam{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: dataURL, Detail: "auto"},
				},
			})
		}
	}
	return openai.ChatCompletionUserMessageParamContentUnion{OfArrayOfContentParts: arr}
}

func joinedText(parts []Part) string {
	var s string
	for _, p := range parts {
		if t, ok := p.(Text); ok {
			if s != "" {
				s += "\n"
			}
			s += t.Value
		}
	}
	return s
}

func flattenToolResults(parts []Part) []Part {
	out := make([]Part, 0, len(parts))
	for _, p := range parts {
		if tr, ok := p.(ToolResult); ok {
			out = append(out, tr.Content...)
		} else {
			out = append(out, p)
		}
	}
	return out
}

func buildOpenAITools(tools []Tool) []openai.ChatCompletionToolParam {
	out := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, t := range tools {
		fd := sharedFunctionDefinition(t)
		out = append(out, openai.ChatCompletionToolParam{
			Function: fd,
			Type:     "function",
		})
	}
	return out
}

func sharedFunctionDefinition(t Tool) shared.FunctionDefinitionParam {
	fd := shared.FunctionDefinitionParam{
		Name:       t.Name,
		Parameters: shared.FunctionParameters(t.Parameters),
	}
	if t.Description != "" {
		fd.Description = openai.String(t.Description)
	}
	return fd
}

func buildOpenAIToolChoice(tc ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch tc.Mode {
	case ToolChoiceAuto:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}
	case ToolChoiceNone:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("none")}
	case ToolChoiceRequired:
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}
	case ToolChoiceSpecific:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfChatCompletionNamedToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: tc.Name},
				Type:     "function",
			},
		}
	}
	return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}
}

func buildOpenAIResponseFormat(rf *ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	switch rf.Type {
	case "json_object":
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{Type: "json_object"},
		}
	case "json_schema":
		if rf.JSONSchema == nil {
			return openai.ChatCompletionNewParamsResponseFormatUnion{}
		}
		p := shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   rf.JSONSchema.Name,
				Schema: rf.JSONSchema.Schema,
			},
			Type: "json_schema",
		}
		if rf.JSONSchema.Strict {
			p.JSONSchema.Strict = openai.Bool(true)
		}
		return openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &p}
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{}
}

// --- response conversion (non-streaming) ---

func convertChatCompletion(raw *openai.ChatCompletion) (*Response, error) {
	if raw == nil {
		return nil, fmt.Errorf("nil chat completion response")
	}
	if len(raw.Choices) == 0 {
		return &Response{
			ID:      raw.ID,
			Model:   raw.Model,
			Usage:   convertUsage(raw.Usage),
			Raw:     raw,
			Message: Message{Role: RoleAssistant},
		}, nil
	}
	ch := raw.Choices[0]
	msg := Message{Role: RoleAssistant}
	if ch.Message.Content != "" {
		msg.Content = []Part{Text{Value: ch.Message.Content}}
	}
	if len(ch.Message.ToolCalls) > 0 {
		for _, tc := range ch.Message.ToolCalls {
			if tc.Type != "function" {
				continue
			}
			args := tc.Function.Arguments
			if args == "" {
				args = "{}"
			}
			// validate JSON
			if !json.Valid([]byte(args)) {
				args = "null"
			}
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: args,
			})
		}
	}
	return &Response{
		ID:         raw.ID,
		Model:      raw.Model,
		Message:    msg,
		StopReason: ch.FinishReason,
		Usage:      convertUsage(raw.Usage),
		Raw:        raw,
	}, nil
}

func convertUsage(u openai.CompletionUsage) Usage {
	return Usage{
		InputTokens:      int(u.PromptTokens),
		OutputTokens:     int(u.CompletionTokens),
		ReasoningTokens:  int(u.CompletionTokensDetails.ReasoningTokens),
		CacheReadTokens:  int(u.PromptTokensDetails.CachedTokens),
		CacheWriteTokens: 0, // chat-completions wire doesn't differentiate cache write from prompt here
		TotalTokens:      int(u.TotalTokens),
	}
}

// --- response conversion (streaming) ---

func convertChatChunk(chunk openai.ChatCompletionChunk) (Event, bool, error) {
	if chunk.ID == "" && len(chunk.Choices) == 0 {
		return Event{}, false, nil
	}
	if len(chunk.Choices) == 0 {
		// empty choice can mean usage-only delta or stream-end; emit a Done
		// with usage if available.
		if chunk.Usage.TotalTokens > 0 || chunk.Usage.PromptTokens > 0 {
			u := convertUsage(chunk.Usage)
			return Event{Type: EventDone, Usage: &u, StopReason: "stop"}, true, nil
		}
		return Event{}, false, nil
	}
	ch := chunk.Choices[0]

	if ch.FinishReason != "" {
		u := convertUsage(chunk.Usage)
		return Event{
			Type:       EventDone,
			StopReason: ch.FinishReason,
			Usage:      &u,
		}, true, nil
	}

	if ch.Delta.Content != "" {
		return Event{
			Type:      EventContentDelta,
			TextDelta: ch.Delta.Content,
		}, true, nil
	}
	for _, tc := range ch.Delta.ToolCalls {
		if tc.Type != "" && tc.Type != "function" {
			continue
		}
		if tc.ID != "" && tc.Function.Name != "" {
			return Event{
				Type: EventToolCallStart,
				ToolCall: &ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
				},
			}, true, nil
		}
		if tc.Function.Arguments != "" {
			return Event{
				Type:          EventToolCallDelta,
				ArgumentDelta: tc.Function.Arguments,
			}, true, nil
		}
	}
	return Event{}, false, nil
}
