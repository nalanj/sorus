package basis

// Reasoning configures model reasoning effort/budget.
//
// Each implementation reads only the fields it understands; over-populating
// is harmless. For chat-completions providers only Effort is honored today
// (it maps to the upstream `reasoning_effort` parameter).
type Reasoning struct {
	// Effort is a categorical effort level: "low", "medium", "high",
	// "xhigh", etc. Provider-specific values are accepted verbatim.
	Effort string

	// BudgetTokens is the reasoning token budget. Zero means unset.
	// Honored by providers that expose a budget-style control
	// (Anthropic `thinking.budget_tokens`, Gemini `thinking_budget`,
	// DeepSeek `thinking_budget`); ignored by chat-completions-style
	// providers that don't take a budget.
	BudgetTokens int
}
