package sorus

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestBuildAnthropicParamsMinimal(t *testing.T) {
	m := &Model{ID: "anthropic/claude-sonnet-4-5"}
	req := NewRequest(m).User("hi")
	p, err := buildAnthropicParams(req)
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Model) != "anthropic/claude-sonnet-4-5" {
		t.Errorf("model mismatch: %s", p.Model)
	}
	if p.MaxTokens == 0 {
		t.Errorf("anthropic requires MaxTokens; default should be set")
	}
	if len(p.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(p.Messages))
	}
}

func TestBuildAnthropicParamsSystemPulledOut(t *testing.T) {
	m := &Model{ID: "anthropic/claude-sonnet-4-5"}
	req := NewRequest(m,
		SystemMessage("be brief"),
		UserMessage("hi"),
	)
	p, err := buildAnthropicParams(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.System) != 1 {
		t.Errorf("expected 1 system block, got %d", len(p.System))
	}
	if len(p.Messages) != 1 {
		t.Errorf("expected 1 message in messages, got %d (system should be at top level)", len(p.Messages))
	}
}

func TestBuildAnthropicParamsAllFields(t *testing.T) {
	m := &Model{ID: "anthropic/claude-sonnet-4-5"}
	req := NewRequest(m,
		SystemMessage("be helpful"),
		UserMessage("hi"),
	).
		Temperature(0.6).
		MaxTokens(2048).
		TopP(0.9).
		Stop("STOP").
		Reasoning(Reasoning{BudgetTokens: 1024}).
		Tools(Tool{
			Name:        "echo",
			Description: "echo input",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
				"required": []any{"text"},
			},
		}).
		ToolChoice(ToolChoice{Mode: ToolChoiceRequired})

	p, err := buildAnthropicParams(req)
	if err != nil {
		t.Fatal(err)
	}
	if p.MaxTokens != 2048 {
		t.Errorf("max tokens: %d", p.MaxTokens)
	}
	s := mustMarshal(t, p)
	wants := []string{
		`"model":"anthropic/claude-sonnet-4-5"`,
		`"max_tokens":2048`,
		`"temperature":0.6`,
		`"top_p":0.9`,
		`"stop_sequences":["STOP"]`,
		`"be helpful"`,
		`"hi"`,
		`"echo"`,
		`"echo input"`,
		`"thinking"`,
		`"budget_tokens":1024`,
		`"any"`, // tool_choice required
	}
	for _, w := range wants {
		if !strings.Contains(s, w) {
			t.Errorf("missing %q in:\n%s", w, s)
		}
	}
}

func TestBuildAnthropicToolChoice(t *testing.T) {
	cases := []struct {
		name        string
		mode        ToolChoiceMode
		toolName    string
		wantPresent []string
	}{
		{"auto", ToolChoiceAuto, "", []string{`"auto"`}},
		{"none", ToolChoiceNone, "", []string{`"none"`}},
		{"required", ToolChoiceRequired, "", []string{`"any"`}},
		{"specific", ToolChoiceSpecific, "echo", []string{`"tool"`, `"name":"echo"`}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := buildAnthropicToolChoice(ToolChoice{Mode: c.mode, Name: c.toolName})
			s := mustMarshal(t, v)
			for _, w := range c.wantPresent {
				if !strings.Contains(s, w) {
					t.Errorf("missing %q in %s", w, s)
				}
			}
		})
	}
}

func TestBuildAnthropicTools(t *testing.T) {
	tools := buildAnthropicTools([]Tool{{
		Name:        "search",
		Description: "search stuff",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"q": map[string]any{"type": "string"},
			},
			"required": []any{"q"},
		},
	}})
	if len(tools) != 1 {
		t.Fatalf("got %d tools", len(tools))
	}
	if tools[0].OfTool == nil {
		t.Fatal("OfTool nil")
	}
	if tools[0].OfTool.Name != "search" {
		t.Errorf("name: %s", tools[0].OfTool.Name)
	}
	if tools[0].OfTool.InputSchema.Type != "object" {
		t.Errorf("type: %s", tools[0].OfTool.InputSchema.Type)
	}
	if len(tools[0].OfTool.InputSchema.Required) != 1 || tools[0].OfTool.InputSchema.Required[0] != "q" {
		t.Errorf("required: %v", tools[0].OfTool.InputSchema.Required)
	}
}

func TestBuildAnthropicToolsRequiredAsStrings(t *testing.T) {
	tools := buildAnthropicTools([]Tool{{
		Name: "f",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"a", "b"},
		},
	}})
	if got := tools[0].OfTool.InputSchema.Required; len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("required: %v", got)
	}
}

func TestAnthropicMediaType(t *testing.T) {
	cases := []struct {
		in   string
		want anthropic.Base64ImageSourceMediaType
		ok   bool
	}{
		{"image/jpeg", anthropic.Base64ImageSourceMediaTypeImageJPEG, true},
		{"image/jpg", anthropic.Base64ImageSourceMediaTypeImageJPEG, true},
		{"image/png", anthropic.Base64ImageSourceMediaTypeImagePNG, true},
		{"image/gif", anthropic.Base64ImageSourceMediaTypeImageGIF, true},
		{"image/webp", anthropic.Base64ImageSourceMediaTypeImageWebP, true},
		{"image/avif", "", false},
		{"application/pdf", "", false},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			got, err := anthropicMediaType(c.in)
			if c.ok && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !c.ok && err == nil {
				t.Fatalf("expected error, got %q", got)
			}
			if c.ok && got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestConvertAnthropicMessageText(t *testing.T) {
	resp, err := convertAnthropicMessage(mustParseAnthropicMessage(t, `{
		"id": "msg_1",
		"model": "claude-sonnet-4-5",
		"content": [{"type": "text", "text": "Hello!"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 100, "output_tokens": 50}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "msg_1" {
		t.Errorf("id: %s", resp.ID)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason: %s", resp.StopReason)
	}
	if len(resp.Message.Content) != 1 {
		t.Fatalf("expected 1 content part")
	}
	txt, ok := resp.Message.Content[0].(Text)
	if !ok || txt.Value != "Hello!" {
		t.Errorf("content: %+v", resp.Message.Content[0])
	}
	if resp.Usage.InputTokens != 100 {
		t.Errorf("input tokens: %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 50 {
		t.Errorf("output tokens: %d", resp.Usage.OutputTokens)
	}
}

func TestConvertAnthropicMessageToolUse(t *testing.T) {
	resp, err := convertAnthropicMessage(mustParseAnthropicMessage(t, `{
		"id": "msg_2",
		"model": "claude-sonnet-4-5",
		"content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "NYC"}}],
		"stop_reason": "tool_use"
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call")
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "tu_1" {
		t.Errorf("id: %s", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("name: %s", tc.Name)
	}
	// Compare semantically, since re-serialization may change whitespace.
	var got, want map[string]any
	if err := json.Unmarshal([]byte(tc.Arguments), &got); err != nil {
		t.Fatalf("args not valid JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(`{"location":"NYC"}`), &want); err != nil {
		t.Fatal(err)
	}
	if got["location"] != want["location"] {
		t.Errorf("args: got %s", tc.Arguments)
	}
}

func TestConvertAnthropicMessagePassThroughEmptyArgs(t *testing.T) {
	// input omitted: Input field stays nil byte, we substitute "{}".
	resp, err := convertAnthropicMessage(mustParseAnthropicMessage(t, `{
		"id": "x",
		"content": [{"type": "tool_use", "id": "tu_1", "name": "fn"}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.ToolCalls[0].Arguments != "{}" {
		t.Errorf("expected {} for empty input, got %q", resp.Message.ToolCalls[0].Arguments)
	}
}

func mustParseAnthropicMessage(t *testing.T, body string) *anthropic.Message {
	t.Helper()
	var m anthropic.Message
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("parse anthropic message: %v", err)
	}
	return &m
}

func TestConvertAnthropicStreamEventTextDelta(t *testing.T) {
	ev := anthropic.MessageStreamEventUnion{
		Type: "content_block_delta",
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type: "text_delta",
			Text: "Hi",
		},
	}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventContentDelta || out.TextDelta != "Hi" {
		t.Errorf("got %+v", out)
	}
}

func TestConvertAnthropicStreamEventThinkingDelta(t *testing.T) {
	ev := anthropic.MessageStreamEventUnion{
		Type: "content_block_delta",
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:     "thinking_delta",
			Thinking: "pondering...",
		},
	}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventReasoningDelta || out.ReasoningDelta != "pondering..." {
		t.Errorf("got %+v", out)
	}
}

func TestConvertAnthropicStreamEventInputJSONDelta(t *testing.T) {
	ev := anthropic.MessageStreamEventUnion{
		Type: "content_block_delta",
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:       "input_json_delta",
			PartialJSON: `{"q":"ny`,
		},
	}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventToolCallDelta || out.ArgumentDelta != `{"q":"ny` {
		t.Errorf("got %+v", out)
	}
}

func TestConvertAnthropicStreamEventToolUseBlockStart(t *testing.T) {
	var ev anthropic.MessageStreamEventUnion
	if err := json.Unmarshal([]byte(`{
		"type": "content_block_start",
		"index": 1,
		"content_block": {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}
	}`), &ev); err != nil {
		t.Fatal(err)
	}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventToolCallStart {
		t.Errorf("type: %s", out.Type)
	}
	if out.ToolCall == nil || out.ToolCall.ID != "tu_1" || out.ToolCall.Name != "search" {
		t.Errorf("tool call: %+v", out.ToolCall)
	}
}

func TestConvertAnthropicStreamEventMessageDelta(t *testing.T) {
	ev := anthropic.MessageStreamEventUnion{
		Type: "message_delta",
		Delta: anthropic.MessageStreamEventUnionDelta{
			StopReason: "end_turn",
		},
		Usage: anthropic.MessageDeltaUsage{
			InputTokens:              10,
			OutputTokens:             20,
			CacheReadInputTokens:     5,
			CacheCreationInputTokens: 7,
		},
	}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventDone {
		t.Errorf("type: %s", out.Type)
	}
	if out.StopReason != "end_turn" {
		t.Errorf("stop reason: %s", out.StopReason)
	}
	if out.Usage == nil {
		t.Fatal("usage nil")
	}
	if out.Usage.InputTokens != 10 || out.Usage.OutputTokens != 20 {
		t.Errorf("usage mismatch: %+v", out.Usage)
	}
	if out.Usage.CacheReadTokens != 5 || out.Usage.CacheWriteTokens != 7 {
		t.Errorf("cache tokens wrong: %+v", out.Usage)
	}
}

func TestConvertAnthropicStreamEventMessageStart(t *testing.T) {
	ev := anthropic.MessageStreamEventUnion{Type: "message_start"}
	out, ok, err := convertAnthropicStreamEvent(ev)
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if out.Type != EventContentStart {
		t.Errorf("type: %s", out.Type)
	}
}

func TestConvertAnthropicStreamEventSwallowableTypes(t *testing.T) {
	types := []string{"ping", "message_stop", "content_block_stop", "error", "compact_message"}
	for _, tName := range types {
		t.Run(tName, func(t *testing.T) {
			ev := anthropic.MessageStreamEventUnion{Type: tName}
			_, ok, err := convertAnthropicStreamEvent(ev)
			if err != nil {
				t.Fatal(err)
			}
			if ok {
				t.Errorf("expected ok=false for %q", tName)
			}
		})
	}
}
