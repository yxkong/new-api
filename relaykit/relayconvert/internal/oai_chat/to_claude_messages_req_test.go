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

func TestOpenAIChatRequestToClaudeMessagesMergesAdjacentAssistantToolCalls(t *testing.T) {
	toolCall := func(id string) []dto.ToolCallRequest {
		return []dto.ToolCallRequest{{
			ID:       id,
			Type:     "function",
			Function: dto.FunctionRequest{Name: "lookup_" + id, Arguments: `{"q":"` + id + `"}`},
		}}
	}
	assistant := func(content any, toolCalls []dto.ToolCallRequest) dto.Message {
		message := dto.Message{Role: "assistant", Content: content}
		if toolCalls != nil {
			message.SetToolCalls(toolCalls)
		}
		return message
	}

	tests := []struct {
		name         string
		messages     []dto.Message
		wantTexts    []string
		wantToolUses []string
	}{
		{
			name: "both assistants have tool calls",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a and b."},
				assistant("", toolCall("call_a")),
				assistant("", toolCall("call_b")),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
				{Role: "tool", ToolCallId: "call_b", Content: "result b"},
			},
			wantTexts:    []string{"..."},
			wantToolUses: []string{"call_a", "call_b"},
		},
		{
			name: "tool calls followed by plain text",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a."},
				assistant(nil, toolCall("call_a")),
				assistant("Checking a.", nil),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
			},
			wantTexts:    []string{"Checking a."},
			wantToolUses: []string{"call_a"},
		},
		{
			name: "literal ellipsis with tool calls is preserved",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a and b."},
				assistant("...", toolCall("call_a")),
				assistant("Checking b.", toolCall("call_b")),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
				{Role: "tool", ToolCallId: "call_b", Content: "result b"},
			},
			wantTexts:    []string{"... Checking b."},
			wantToolUses: []string{"call_a", "call_b"},
		},
		{
			name: "literal ellipsis replacing a placeholder survives the next merge",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a."},
				assistant(nil, toolCall("call_a")),
				assistant("...", nil),
				assistant("Checking a.", nil),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
			},
			wantTexts:    []string{"... Checking a."},
			wantToolUses: []string{"call_a"},
		},
		{
			name: "literal ellipsis before tool calls survives repeated merges",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a."},
				assistant("...", nil),
				assistant("", toolCall("call_a")),
				assistant("Checking a.", nil),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
			},
			wantTexts:    []string{"... Checking a."},
			wantToolUses: []string{"call_a"},
		},
		{
			name: "three tool-only assistants keep one placeholder",
			messages: []dto.Message{
				{Role: "user", Content: "Look up a, b and c."},
				assistant(nil, toolCall("call_a")),
				assistant("", toolCall("call_b")),
				assistant("", toolCall("call_c")),
				{Role: "tool", ToolCallId: "call_a", Content: "result a"},
				{Role: "tool", ToolCallId: "call_b", Content: "result b"},
				{Role: "tool", ToolCallId: "call_c", Content: "result c"},
			},
			wantTexts:    []string{"..."},
			wantToolUses: []string{"call_a", "call_b", "call_c"},
		},
		{
			name: "both assistants are plain text",
			messages: []dto.Message{
				{Role: "user", Content: "Hello."},
				assistant("First.", nil),
				assistant("Second.", nil),
			},
			wantTexts: []string{"First. Second."},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxTokens := uint(1024)
			got, err := OpenAIChatRequestToClaudeMessages(context.Background(), nil, dto.GeneralOpenAIRequest{
				Model:     "claude-test",
				MaxTokens: &maxTokens,
				Messages:  tt.messages,
			})
			require.NoError(t, err)

			var wantToolResults []dto.ClaudeMediaMessage
			for _, message := range tt.messages {
				if message.Role == "tool" {
					wantToolResults = append(wantToolResults, dto.ClaudeMediaMessage{
						Type:      "tool_result",
						ToolUseId: message.ToolCallId,
						Content:   message.Content,
					})
				}
			}
			if len(wantToolResults) > 0 {
				require.Len(t, got.Messages, 3)
				assert.Equal(t, "user", got.Messages[2].Role)
				assert.Equal(t, wantToolResults, got.Messages[2].Content)
			} else {
				require.Len(t, got.Messages, 2)
			}
			assert.Equal(t, "assistant", got.Messages[1].Role)

			var texts, toolUses []string
			switch content := got.Messages[1].Content.(type) {
			case string:
				texts = append(texts, content)
			case []dto.ClaudeMediaMessage:
				for _, block := range content {
					switch block.Type {
					case "text":
						texts = append(texts, block.GetText())
					case "tool_use":
						toolUses = append(toolUses, block.Id)
						assert.Equal(t, "lookup_"+block.Id, block.Name)
						assert.Equal(t, map[string]any{"q": block.Id}, block.Input)
					}
				}
			default:
				require.Failf(t, "unexpected assistant content type", "%T", content)
			}
			assert.Equal(t, tt.wantTexts, texts)
			assert.Equal(t, tt.wantToolUses, toolUses)
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
