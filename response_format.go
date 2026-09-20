package sorus

// ResponseFormat asks the model to produce structured output.
//
// Type is "json_object" for unconstrained JSON output, or "json_schema"
// to constrain by a schema. JSONSchema is required when Type ==
// "json_schema".
type ResponseFormat struct {
	Type       string      // "json_object" | "json_schema"
	JSONSchema *JSONSchema // required when Type == "json_schema"
}

// JSONSchema is the JSON Schema for the response. Schema is the schema body
// as a Go map. Use a schema-builder library like invopop/jsonschema if you
// want typesafety.
type JSONSchema struct {
	Name   string
	Schema map[string]any
	Strict bool // best-effort "strict" mode; provider-dependent whether honored
}
