package sorus

import (
	"testing"
)

func TestRequestBuilder(t *testing.T) {
	// We just need a model ref to bind the request; the catalog lookup is
	// not exercised here.
	m := &Model{ID: "openai/gpt-5-nano", Name: "GPT-5 Nano"}

	r := NewRequest(m).
		System("you are concise").
		Temperature(0.5).
		User("hi").
		MaxTokens(100).
		Tools(Tool{Name: "echo", Parameters: map[string]any{"type": "object"}}).
		Reasoning(Reasoning{Effort: "low"})

	if r.Model().ID != "openai/gpt-5-nano" {
		t.Errorf("model id mismatch")
	}
	if got := r.temperaturePtr(); got == nil || *got != 0.5 {
		t.Errorf("temperature: %v", got)
	}
	if got := r.maxTokensPtr(); got == nil || *got != 100 {
		t.Errorf("max tokens: %v", got)
	}
	if got := r.reasoningPtr(); got == nil || got.Effort != "low" {
		t.Errorf("reasoning: %v", got)
	}
	if tools := r.tools; len(tools) != 1 || tools[0].Name != "echo" {
		t.Errorf("tools: %v", tools)
	}

	// Clone independence.
	r2 := r.Clone().MaxTokens(200)
	if got := r.maxTokensPtr(); got == nil || *got != 100 {
		t.Errorf("original mutated: %v", got)
	}
	if got := r2.maxTokensPtr(); got == nil || *got != 200 {
		t.Errorf("clone didn't change: %v", got)
	}
}

func TestMessageConstructors(t *testing.T) {
	if SystemMessage("s").Role != RoleSystem {
		t.Errorf("system role")
	}
	if UserMessage("u").Content[0].(Text).Value != "u" {
		t.Errorf("user content")
	}
	if AssistantMessage("a").Role != RoleAssistant {
		t.Errorf("assistant role")
	}
}
