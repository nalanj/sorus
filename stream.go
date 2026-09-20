package basis

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// Stream is a streaming response. Iterate with [Stream.Next], inspect the
// latest event with [Stream.Event], then call [Stream.Close]. Read errors
// via [Stream.Err].
type Stream struct {
	ctx     context.Context
	current Event
	err     error
	done    bool
	closed  bool

	next func() (Event, bool, error)
}

// Next advances the stream. Returns false when the stream is exhausted or
// closed. After Next returns false, [Stream.Err] reports any terminal
// error.
func (s *Stream) Next() bool {
	if s.closed || s.done {
		return false
	}
	ev, ok, err := s.next()
	if err != nil {
		s.err = err
		s.done = true
		return false
	}
	if !ok {
		s.done = true
		return false
	}
	s.current = ev
	if ev.Type == EventDone || ev.Type == EventError {
		s.done = true
		if ev.Type == EventError {
			s.err = ev.Err
		}
	}
	return true
}

// Event returns the most recent event from the most recent successful
// [Stream.Next] call.
func (s *Stream) Event() Event { return s.current }

// Err returns the terminal error from the stream, if any.
func (s *Stream) Err() error { return s.err }

// Close releases the stream's underlying resources. Calling Close after
// Close is a no-op.
func (s *Stream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	if s.done {
		return s.err
	}
	s.done = true
	return nil
}

// Collect drains the stream into a [Response].
func (s *Stream) Collect() (*Response, error) {
	if s.closed {
		return nil, ErrStreamClosed
	}
	defer s.Close()

	var (
		content       string
		reasoning     string
		usage         *Usage
		stopReason    string
		toolCalls     []ToolCall
		finalMessage  *Message
	)
	for s.Next() {
		ev := s.Event()
		switch ev.Type {
		case EventContentDelta:
			content += ev.TextDelta
		case EventReasoningDelta:
			reasoning += ev.ReasoningDelta
		case EventToolCallStart:
			if ev.ToolCall != nil {
				tc := *ev.ToolCall
				tc.Arguments = ""
				toolCalls = append(toolCalls, tc)
			}
		case EventToolCallDelta:
			if len(toolCalls) > 0 {
				toolCalls[len(toolCalls)-1].Arguments += ev.ArgumentDelta
			}
		case EventToolCallEnd:
			// nothing — the assembled ToolCall is already in toolCalls
		case EventDone:
			usage = ev.Usage
			stopReason = ev.StopReason
			if ev.Message != nil {
				finalMessage = ev.Message
			}
		case EventError:
			return nil, ev.Err
		}
	}
	if err := s.Err(); err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	if finalMessage == nil {
		// Build a synthetic message from collected fragments.
		m := Message{Role: RoleAssistant}
		if content != "" {
			m.Content = append(m.Content, Text{Value: content})
		}
		if reasoning != "" {
			m.Content = append(m.Content, Text{Value: reasoning})
		}
		if len(toolCalls) > 0 {
			m.ToolCalls = toolCalls
		}
		finalMessage = &m
	}
	return &Response{
		Message:    *finalMessage,
		StopReason: stopReason,
		Usage:      derefUsage(usage),
	}, nil
}

func derefUsage(u *Usage) Usage {
	if u == nil {
		return Usage{}
	}
	return *u
}

// wrapStream builds a Stream from a per-event callback. The callback should
// return (Event, true, nil) for each event, and (Event{}, false, nil) when
// the stream is exhausted. On error: (Event{Type:EventError, Err: err},
// false, err) — both the error value and the err return are returned.
//
// This helper is used by Client.Stream implementations.
func wrapStream(next func() (Event, bool, error)) *Stream {
	return &Stream{next: next}
}

// IDForError builds an [Event] of type [EventError] from an error.
func errorEvent(err error) Event {
	if err == nil {
		err = errors.New("stream error")
	}
	return Event{Type: EventError, Err: fmt.Errorf("stream: %w", err)}
}

// EventType identifies the kind of [Event].
type EventType string

const (
	EventContentStart   EventType = "content_start"
	EventContentDelta   EventType = "content_delta"
	EventContentEnd     EventType = "content_end"
	EventReasoningStart EventType = "reasoning_start"
	EventReasoningDelta EventType = "reasoning_delta"
	EventReasoningEnd   EventType = "reasoning_end"
	EventToolCallStart  EventType = "tool_call_start"
	EventToolCallDelta  EventType = "tool_call_delta"
	EventToolCallEnd    EventType = "tool_call_end"
	EventDone           EventType = "done"
	EventError          EventType = "error"
)

// Event is a single event yielded by a [Stream].
type Event struct {
	Type EventType

	// EventContentDelta:
	TextDelta string

	// EventReasoningDelta:
	ReasoningDelta string

	// EventToolCallStart: the assembled tool call with ID and Name; Arguments
	// may be empty if the model streams them separately.
	ToolCall *ToolCall

	// EventToolCallDelta: incremental JSON fragment of the in-progress
	// arguments.
	ArgumentDelta string

	// EventDone: aggregated terminal fields.
	StopReason string
	Usage      *Usage
	Message    *Message

	// EventError:
	Err error
}

// ToolCall is one tool call emitted by the model. Arguments is a JSON
// string; json.Unmarshal it to get structured input.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}
