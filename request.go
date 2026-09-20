package sorus

// Request is the unit of work sent to a [Client]. It is built by fluent
// configuration starting with [NewRequest].
//
// Internally, all optional fields are stored as pointers (or sentinel zero
// values) so we can distinguish "unset" from "set to zero". Externally,
// callers see only literal setters taking primitive types.
type Request struct {
	model          *Model
	messages       []Message
	tools          []Tool
	toolChoice     *ToolChoice
	temperature    *float64
	maxTokens      *int
	topP           *float64
	stop           []string
	reasoning      *Reasoning
	responseFormat *ResponseFormat
}

// NewRequest begins a fluent request bound to model. Initial messages can
// be passed positionally.
func NewRequest(model *Model, messages ...Message) *Request {
	return &Request{
		model:    model,
		messages: append([]Message(nil), messages...),
	}
}

// Model returns the bound model.
func (r *Request) Model() *Model { return r.model }

// Messages returns the current message list (a copy).
func (r *Request) Messages() []Message {
	out := make([]Message, len(r.messages))
	copy(out, r.messages)
	return out
}

// Message appends a complete [Message] to the conversation.
func (r *Request) Message(msg Message) *Request {
	r.messages = append(r.messages, msg)
	return r
}

// System appends a plain-text system message. Equivalent to
// Message(SystemMessage(text)).
func (r *Request) System(text string) *Request {
	return r.Message(SystemMessage(text))
}

// User appends a plain-text user message.
func (r *Request) User(text string) *Request {
	return r.Message(UserMessage(text))
}

// Assistant appends a plain-text assistant message.
func (r *Request) Assistant(text string) *Request {
	return r.Message(AssistantMessage(text))
}

// Tools appends to the tool list.
func (r *Request) Tools(tool ...Tool) *Request {
	r.tools = append(r.tools, tool...)
	return r
}

// ToolChoice sets the tool-choice mode for this request.
func (r *Request) ToolChoice(choice ToolChoice) *Request {
	r.toolChoice = &choice
	return r
}

// Temperature sets the sampling temperature.
func (r *Request) Temperature(v float64) *Request {
	r.temperature = &v
	return r
}

// MaxTokens sets the max output token count.
func (r *Request) MaxTokens(v int) *Request {
	r.maxTokens = &v
	return r
}

// TopP sets the nucleus sampling parameter.
func (r *Request) TopP(v float64) *Request {
	r.topP = &v
	return r
}

// Stop appends to the stop-sequence list.
func (r *Request) Stop(seqs ...string) *Request {
	r.stop = append(r.stop, seqs...)
	return r
}

// Reasoning sets the reasoning controls.
func (r *Request) Reasoning(r2 Reasoning) *Request {
	r.reasoning = &r2
	return r
}

// ResponseFormat sets the structured-output format.
func (r *Request) ResponseFormat(rf *ResponseFormat) *Request {
	r.responseFormat = rf
	return r
}

// Clone returns an independent copy of r. Mutations on the clone do not
// affect the original.
func (r *Request) Clone() *Request {
	cp := *r
	cp.messages = append([]Message(nil), r.messages...)
	cp.tools = append([]Tool(nil), r.tools...)
	cp.stop = append([]string(nil), r.stop...)
	if r.toolChoice != nil {
		t := *r.toolChoice
		cp.toolChoice = &t
	}
	if r.temperature != nil {
		v := *r.temperature
		cp.temperature = &v
	}
	if r.maxTokens != nil {
		v := *r.maxTokens
		cp.maxTokens = &v
	}
	if r.topP != nil {
		v := *r.topP
		cp.topP = &v
	}
	if r.reasoning != nil {
		v := *r.reasoning
		cp.reasoning = &v
	}
	// *ResponseFormat is shared with the original — copy if mutated.
	if r.responseFormat != nil {
		v := *r.responseFormat
		if v.JSONSchema != nil {
			s := *v.JSONSchema
			if s.Schema != nil {
				s.Schema = cloneAnyMap(s.Schema)
			}
			v.JSONSchema = &s
		}
		cp.responseFormat = &v
	}
	return &cp
}

func cloneAnyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		switch vv := v.(type) {
		case map[string]any:
			out[k] = cloneAnyMap(vv)
		default:
			out[k] = v
		}
	}
	return out
}

// Internal accessors used by client implementations. These return zero values
// plus ok=false for unset fields.

func (r *Request) temperaturePtr() *float64 { return r.temperature }
func (r *Request) maxTokensPtr() *int       { return r.maxTokens }
func (r *Request) topPPtr() *float64        { return r.topP }
func (r *Request) stopSeqs() []string       { return r.stop }
func (r *Request) toolChoicePtr() *ToolChoice { return r.toolChoice }
func (r *Request) reasoningPtr() *Reasoning  { return r.reasoning }
func (r *Request) responseFormatPtr() *ResponseFormat { return r.responseFormat }
