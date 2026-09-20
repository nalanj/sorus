package basis

// Tool describes one tool the model may call. Parameters is a JSON Schema
// object; build it manually or with invopop/jsonschema.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// ToolChoice controls how the model picks tools. Mode is required; Name is
// only meaningful when Mode == ToolChoiceSpecific.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// ToolChoiceMode enumerates the tool-selection modes.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceSpecific ToolChoiceMode = "specific"
)
