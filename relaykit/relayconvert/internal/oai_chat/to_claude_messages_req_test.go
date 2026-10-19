package oaichat

import (
	"context"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIChatRequestToClaudeMessagesNormalizesToolInputSchema(t *testing.T) {
	tests := []struct {
		name       string
		parameters any
		wantSchema map[string]any
	}{
		{
			name:       "omitted parameters",
			parameters: nil,
			wantSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		{
			name: "missing type and properties",
			parameters: map[string]any{
				"additionalProperties": false,
			},
			wantSchema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"additionalProperties": false,
			},
		},
		{
			name: "non-string type",
			parameters: map[string]any{
				"type":       123,
				"properties": map[string]any{},
			},
			wantSchema: map[string]any{
				"type":       123,
				"properties": map[string]any{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxTokens := uint(1024)
			got, err := OpenAIChatRequestToClaudeMessages(context.Background(), nil, dto.GeneralOpenAIRequest{
				Model:     "claude-test",
				MaxTokens: &maxTokens,
				Messages: []dto.Message{
					{Role: "user", Content: "Call the tool."},
				},
				Tools: []dto.ToolCallRequest{
					{
						Type: "function",
						Function: dto.FunctionRequest{
							Name:        "get_current_time",
							Description: "Get the current time",
							Parameters:  tt.parameters,
						},
					},
				},
			})

			require.NoError(t, err)
			tools, ok := got.Tools.([]any)
			require.True(t, ok)
			require.Len(t, tools, 1)
			tool, ok := tools[0].(*dto.Tool)
			require.True(t, ok)
			assert.Equal(t, "get_current_time", tool.Name)
			assert.Equal(t, tt.wantSchema, tool.InputSchema)
		})
	}
}

func TestOpenAIChatRequestToClaudeMessagesStop(t *testing.T) {
	var jsonStop any
	require.NoError(t, kitutil.Unmarshal([]byte(`["END",1,null,""]`), &jsonStop))

	maxTokens := uint(16)
	tests := []struct {
		name string
		stop any
		want []string
	}{
		{name: "string", stop: "END", want: []string{"END"}},
		{name: "empty-string", stop: "", want: []string{""}},
		{name: "any-strings", stop: []any{"END", ""}, want: []string{"END", ""}},
		{name: "typed-slice", stop: []string{"END", "STOP"}, want: []string{"END", "STOP"}},
		{name: "typed-slice-one", stop: []string{"END"}, want: []string{"END"}},
		{name: "json-array", stop: jsonStop, want: []string{"END", ""}},
		{name: "go-int", stop: []any{"END", 1}, want: []string{"END"}},
		{name: "mixed-non-string", stop: []any{"END", 1, false, nil}, want: []string{"END"}},
		{name: "only-non-string", stop: []any{1, nil}, want: []string{}},
		{name: "empty-array", stop: []any{}, want: []string{}},
		{name: "nil", stop: nil, want: nil},
		{name: "number", stop: float64(1), want: nil},
		{name: "bool", stop: true, want: nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := dto.GeneralOpenAIRequest{
				Model:     "claude-test",
				MaxTokens: &maxTokens,
				Messages: []dto.Message{
					{Role: "user", Content: "hi"},
				},
				Stop: tt.stop,
			}

			var got *dto.ClaudeRequest
			var err error
			require.NotPanics(t, func() {
				got, err = OpenAIChatRequestToClaudeMessages(context.Background(), nil, req)
			})
			require.NoError(t, err)
			require.Equal(t, tt.want, got.StopSequences)
			if tt.name == "empty-string" {
				raw, marshalErr := kitutil.Marshal(got)
				require.NoError(t, marshalErr)
				assert.Contains(t, string(raw), `"stop_sequences":[""]`)
			}
		})
	}
}
