package sorus

// Reasoning configures model reasoning effort/budget.
//
// Each implementation reads only the fields it understands and sends them
// verbatim; chat-completions providers read only Effort. Valid values differ
// per model; see [Model.ReasoningOptions].
type Reasoning struct {
	// Effort is a categorical effort level: "low", "medium", "high",
	// "xhigh", etc. Provider-specific values are accepted verbatim.
	// Maps to `reasoning_effort` on chat-completions providers and
	// `output_config.effort` on Anthropic, where it does not set `thinking`.
	Effort string

	// BudgetTokens is the reasoning token budget. Zero means unset.
	// Honored by providers that expose a budget-style control
	// (Anthropic `thinking.budget_tokens`, Gemini `thinking_budget`,
	// DeepSeek `thinking_budget`); ignored by chat-completions-style
	// providers that don't take a budget.
	BudgetTokens int
}
