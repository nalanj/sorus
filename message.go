package sorus

// Message is one turn in a conversation.
type Message struct {
	Role     Role
	Content  []Part
	ToolCalls []ToolCall // populated on assistant messages that emit tool calls
}

// Role is the author of a [Message].
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Part is a content part of a [Message]. The interface is sealed: only
// [Text], [ImageURL], [ImageData], and [ToolResult] implement it. Adding
// new part types is purely additive because the marker method is unexported.
type Part interface {
	isPart()
}

// Text is a plain-text content part.
type Text struct {
	Value string
}

// ImageURL is an image referenced by URL.
type ImageURL struct {
	URL    string
	Detail string // "auto" | "low" | "high"; empty means provider default
}

// ImageData is an image embedded as bytes.
type ImageData struct {
	Data     []byte
	MIMEType string // e.g. "image/png"
}

// ToolResult is one tool result in a [Message] with role [RoleTool].
type ToolResult struct {
	ToolCallID string
	Content    []Part
	IsError    bool
}

// isPart implements [Part].
func (Text) isPart()       {}
func (ImageURL) isPart()   {}
func (ImageData) isPart()  {}
func (ToolResult) isPart() {}

// Text returns the concatenation of every [Text] part in the message.
// Non-text parts are ignored.
func (m Message) Text() string {
	var s string
	for _, p := range m.Content {
		if t, ok := p.(Text); ok {
			s += t.Value
		}
	}
	return s
}

// SystemMessage constructs a plain-text system message.
func SystemMessage(text string) Message {
	return Message{Role: RoleSystem, Content: []Part{Text{Value: text}}}
}

// UserMessage constructs a plain-text user message.
func UserMessage(text string) Message {
	return Message{Role: RoleUser, Content: []Part{Text{Value: text}}}
}

// AssistantMessage constructs a plain-text assistant message.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Content: []Part{Text{Value: text}}}
}
