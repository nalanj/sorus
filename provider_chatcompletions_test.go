package sorus

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go"
)

// Helpers ---------------------------------------------------------------------

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

// Build tests -----------------------------------------------------------------

func TestBuildChatParamsMinimal(t *testing.T) {
	m := &Model{ID: "anthropic/claude-sonnet-4-5", Name: "Claude Sonnet 4.5"}
	req := NewRequest(m)
	p, err := buildChatParams(req)
	if err != nil {
		t.Fatalf("buildChatParams: %v", err)
	}
	if string(p.Model) != "anthropic/claude-sonnet-4-5" {
		t.Errorf("model = %q", p.Model)
	}
	if len(p.Messages) != 0 {
		t.Errorf("expected 0 messages, got %d", len(p.Messages))
	}
}

func TestBuildChatParamsAllFields(t *testing.T) {
	m := &Model{ID: "anthropic/claude-sonnet-4-5"}
	req := NewRequest(m,
		SystemMessage("be concise"),
		UserMessage("hi"),
	).
		Temperature(0.5).
		MaxTokens(100).
		TopP(0.9).
		Stop("END", "STOP").
		Reasoning(Reasoning{Effort: "high"}).
		Tools(Tool{
			Name:        "echo",
			Description: "echoes input",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
			},
		}).
		ToolChoice(ToolChoice{Mode: ToolChoiceRequired}).
		ResponseFormat(&ResponseFormat{Type: "json_object"})

	p, err := buildChatParams(req)
	if err != nil {
		t.Fatalf("buildChatParams: %v", err)
	}

	s := mustMarshal(t, p)

	wants := []string{
		`"model":"anthropic/claude-sonnet-4-5"`,
		`"temperature":0.5`,
		`"max_tokens":100`,
		`"top_p":0.9`,
		`"reasoning_effort":"high"`,
		`"echo"`,                         // tool name
		`"echoes input"`,                 // tool description
		`"required"`,                     // tool_choice
		`"json_object"`,                  // response_format
		`"be concise"`,                   // system message
		`"hi"`,                           // user message
		`["END","STOP"]`,                 // stop sequences as array
	}
	for _, w := range wants {
		if !strings.Contains(s, w) {
			t.Errorf("missing %q in serialized params:\n%s", w, s)
		}
	}
}

func TestBuildChatParamsSingleStopSequence(t *testing.T) {
	m := &Model{ID: "x"}
	req := NewRequest(m).Stop("END")
	p, err := buildChatParams(req)
	if err != nil {
		t.Fatal(err)
	}
	s := mustMarshal(t, p)
	if !strings.Contains(s, `"END"`) {
		t.Errorf("single stop sequence not present: %s", s)
	}
}

func TestBuildChatParamsToolChoice(t *testing.T) {
	m := &Model{ID: "x"}
	cases := []struct {
		name        string
		mode        ToolChoiceMode
		toolName    string
		wantPresent []string
	}{
		{"auto", ToolChoiceAuto, "", []string{`"auto"`}},
		{"none", ToolChoiceNone, "", []string{`"none"`}},
		{"required", ToolChoiceRequired, "", []string{`"required"`}},
		{"specific", ToolChoiceSpecific, "echo", []string{`"function"`, `"name":"echo"`}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := NewRequest(m).ToolChoice(ToolChoice{Mode: c.mode, Name: c.toolName})
			p, err := buildChatParams(req)
			if err != nil {
				t.Fatal(err)
			}
			s := mustMarshal(t, p)
			for _, w := range c.wantPresent {
				if !strings.Contains(s, w) {
					t.Errorf("%s: missing %q in %s", c.name, w, s)
				}
			}
		})
	}
}

func TestBuildChatParamsResponseFormatSchema(t *testing.T) {
	m := &Model{ID: "x"}
	req := NewRequest(m).ResponseFormat(&ResponseFormat{
		Type: "json_schema",
		JSONSchema: &JSONSchema{
			Name:   "weather",
			Strict: true,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary": map[string]any{"type": "string"},
				},
			},
		},
	})
	p, err := buildChatParams(req)
	if err != nil {
		t.Fatal(err)
	}
	s := mustMarshal(t, p)
	wants := []string{
		`"json_schema"`,
		`"strict":true`,
		`"name":"weather"`,
		`"summary"`,
	}
	for _, w := range wants {
		if !strings.Contains(s, w) {
			t.Errorf("missing %q in %s", w, s)
		}
	}
}

func TestBuildChatParamsReasoningEffortOmitted(t *testing.T) {
	m := &Model{ID: "x"}
	req := NewRequest(m).Reasoning(Reasoning{BudgetTokens: 4096}) // effort empty
	p, err := buildChatParams(req)
	if err != nil {
		t.Fatal(err)
	}
	s := mustMarshal(t, p)
	if strings.Contains(s, "reasoning_effort") {
		t.Errorf("reasoning_effort should be omitted when Effort empty: %s", s)
	}
}

// Message conversion tests ---------------------------------------------------

func TestBuildOpenAIMessageRoles(t *testing.T) {
	cases := []struct {
		name     string
		msg      Message
		wantRole string
	}{
		{"system", SystemMessage("s"), "system"},
		{"user", UserMessage("u"), "user"},
		{"assistant", AssistantMessage("a"), "assistant"},
		{"tool", Message{Role: RoleTool, Content: []Part{
			ToolResult{ToolCallID: "tc-1", Content: []Part{Text{Value: "ok"}}},
		}}, "tool"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			u, err := buildOpenAIMessage(c.msg)
			if err != nil {
				t.Fatal(err)
			}
			s := mustMarshal(t, u)
			if !strings.Contains(s, `"role":"`+c.wantRole+`"`) {
				t.Errorf("expected role %q in: %s", c.wantRole, s)
			}
		})
	}
}

func TestBuildOpenAIMessageToolID(t *testing.T) {
	msg := Message{Role: RoleTool, Content: []Part{
		ToolResult{ToolCallID: "call_xyz", Content: []Part{Text{Value: "ok"}}},
	}}
	u, err := buildOpenAIMessage(msg)
	if err != nil {
		t.Fatal(err)
	}
	if u.OfTool == nil {
		t.Fatal("OfTool nil")
	}
	if u.OfTool.ToolCallID != "call_xyz" {
		t.Errorf("tool_call_id = %q", u.OfTool.ToolCallID)
	}
}

func TestBuildOpenAIMessageUnknownRole(t *testing.T) {
	_, err := buildOpenAIMessage(Message{Role: "tool", Content: []Part{Text{Value: "x"}}})
	// RoleTool is valid above; use a fake role
	_, err = buildOpenAIMessage(Message{Role: Role("supervisor"), Content: []Part{Text{Value: "x"}}})
	if err == nil {
		t.Error("expected error for unknown role")
	}
}

func TestContentUnionOfMultimodal(t *testing.T) {
	parts := []Part{
		Text{Value: "what is this?"},
		ImageURL{URL: "https://example.com/x.jpg", Detail: "high"},
	}
	u := contentUnionOf(parts)
	s := mustMarshal(t, u)
	for _, want := range []string{
		`"type":"text"`,
		`"text":"what is this?"`,
		`"type":"image_url"`,
		`"https://example.com/x.jpg"`,
		`"high"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}

func TestFlattenToolResults(t *testing.T) {
	parts := []Part{
		ToolResult{ToolCallID: "a", Content: []Part{Text{Value: "x"}}},
		Text{Value: "between"},
		ToolResult{ToolCallID: "b", Content: []Part{Text{Value: "y"}, Text{Value: "z"}}},
	}
	out := flattenToolResults(parts)
	if len(out) != 4 {
		t.Fatalf("flattenToolResults length: got %d, want 4", len(out))
	}
	want := []string{"x", "between", "y", "z"}
	for i, p := range out {
		txt, ok := p.(Text)
		if !ok {
			t.Errorf("part %d: not Text", i)
			continue
		}
		if txt.Value != want[i] {
			t.Errorf("part %d: %q, want %q", i, txt.Value, want[i])
		}
	}
}

// Response conversion tests --------------------------------------------------

func TestConvertChatCompletion(t *testing.T) {
	raw := &openai.ChatCompletion{
		ID:    "test-id",
		Model: "anthropic/claude-sonnet-4-5",
		Choices: []openai.ChatCompletionChoice{{
			FinishReason: "stop",
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: "Hello!",
			},
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}
	resp, err := convertChatCompletion(raw)
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "test-id" {
		t.Errorf("id: %s", resp.ID)
	}
	if resp.Model != "anthropic/claude-sonnet-4-5" {
		t.Errorf("model: %s", resp.Model)
	}
	if resp.StopReason != "stop" {
		t.Errorf("stop reason: %s", resp.StopReason)
	}
	if len(resp.Message.Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(resp.Message.Content))
	}
	if txt, ok := resp.Message.Content[0].(Text); !ok || txt.Value != "Hello!" {
		t.Errorf("content: %+v", resp.Message.Content[0])
	}
	if resp.Message.Role != RoleAssistant {
		t.Errorf("message role: %s", resp.Message.Role)
	}
	if resp.Usage.InputTokens != 100 {
		t.Errorf("input tokens: %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 50 {
		t.Errorf("output tokens: %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 150 {
		t.Errorf("total tokens: %d", resp.Usage.TotalTokens)
	}
}

func TestConvertChatCompletionWithToolCalls(t *testing.T) {
	raw := &openai.ChatCompletion{
		ID: "tc-id",
		Choices: []openai.ChatCompletionChoice{{
			FinishReason: "tool_calls",
			Message: openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"location":"NYC"}`,
					},
				}},
			},
		}},
	}
	resp, err := convertChatCompletion(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("id: %s", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("name: %s", tc.Name)
	}
	if tc.Arguments != `{"location":"NYC"}` {
		t.Errorf("args: %s", tc.Arguments)
	}
}

func TestConvertChatCompletionEmptyChoices(t *testing.T) {
	raw := &openai.ChatCompletion{
		ID:    "x",
		Model: "y",
	}
	resp, err := convertChatCompletion(raw)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Role != RoleAssistant {
		t.Errorf("expected assistant message with empty choices, got %s", resp.Message.Role)
	}
}

func TestConvertChatCompletionBadJSONArgs(t *testing.T) {
	raw := &openai.ChatCompletion{
		ID: "x",
		Choices: []openai.ChatCompletionChoice{{
			FinishReason: "tool_calls",
			Message: openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID:   "tc1",
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "fn",
						Arguments: "not valid json",
					},
				}},
			},
		}},
	}
	resp, err := convertChatCompletion(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("expected 1 tool call")
	}
	if resp.Message.ToolCalls[0].Arguments != "null" {
		t.Errorf("expected null for invalid args, got %q", resp.Message.ToolCalls[0].Arguments)
	}
}

func TestConvertChatCompletionUsageDetails(t *testing.T) {
	raw := &openai.ChatCompletion{
		ID: "x",
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Content: "ok", Role: "assistant"},
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
			CompletionTokensDetails: openai.CompletionUsageCompletionTokensDetails{
				ReasoningTokens: 30,
			},
			PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{
				CachedTokens: 40,
			},
		},
	}
	resp, err := convertChatCompletion(raw)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.ReasoningTokens != 30 {
		t.Errorf("reasoning tokens: %d", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.CacheReadTokens != 40 {
		t.Errorf("cached tokens: %d", resp.Usage.CacheReadTokens)
	}
}

// Chunk conversion tests -----------------------------------------------------

func TestConvertChatChunkContentDelta(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Content: "Hello",
			},
		}},
	}
	ev, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if ev.Type != EventContentDelta {
		t.Errorf("type: %s", ev.Type)
	}
	if ev.TextDelta != "Hello" {
		t.Errorf("delta: %s", ev.TextDelta)
	}
}

func TestConvertChatChunkToolCallStart(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					ID:    "tc_1",
					Type:  "function",
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Name: "search",
					},
				}},
			},
		}},
	}
	ev, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if ev.Type != EventToolCallStart {
		t.Errorf("type: %s", ev.Type)
	}
	if ev.ToolCall == nil {
		t.Fatal("nil tool call")
	}
	if ev.ToolCall.ID != "tc_1" {
		t.Errorf("id: %s", ev.ToolCall.ID)
	}
	if ev.ToolCall.Name != "search" {
		t.Errorf("name: %s", ev.ToolCall.Name)
	}
}

func TestConvertChatChunkToolCallDelta(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Arguments: `{"q":"hello"}`,
					},
				}},
			},
		}},
	}
	ev, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if ev.Type != EventToolCallDelta {
		t.Errorf("type: %s", ev.Type)
	}
	if ev.ArgumentDelta != `{"q":"hello"}` {
		t.Errorf("delta: %s", ev.ArgumentDelta)
	}
}

func TestConvertChatChunkFinishReason(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index:       0,
			FinishReason: "stop",
		}},
	}
	ev, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if ev.Type != EventDone {
		t.Errorf("type: %s", ev.Type)
	}
	if ev.StopReason != "stop" {
		t.Errorf("stop reason: %s", ev.StopReason)
	}
	if ev.Usage == nil {
		t.Error("expected usage on done event")
	}
}

func TestConvertChatChunkUsageOnly(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
	ev, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if ev.Type != EventDone {
		t.Errorf("type: %s", ev.Type)
	}
}

func TestConvertChatChunkSkippableRoleDelta(t *testing.T) {
	chunk := openai.ChatCompletionChunk{
		ID: "x",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Role: "assistant",
			},
		}},
	}
	_, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected ok=false for role-only delta")
	}
}

func TestConvertChatChunkEmpty(t *testing.T) {
	chunk := openai.ChatCompletionChunk{}
	_, ok, err := convertChatChunk(chunk)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected ok=false for empty chunk")
	}
}
