package basis

// Response is what [Client.Chat] returns.
type Response struct {
	ID         string
	Model      string
	Message    Message
	StopReason string // "end_turn" | "max_tokens" | "tool_use" | "stop_sequence" | provider-specific
	Usage      Usage
	Raw        any // provider-specific raw payload; nil if the implementation doesn't expose one
}

// Usage is the model's reported token usage.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	ReasoningTokens  int
	CacheReadTokens  int
	CacheWriteTokens int
	TotalTokens      int
}
