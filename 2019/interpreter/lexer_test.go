package lexer

import (
	"testing"

	"github.com/xxks-kkk/Code-for-blog/interpreter/token"
)

func TestNextToken(t *testing.T) {
	input := `=+(){},;`

	tests := []struct {
		expectedType    token.TokenType
		expectedLiteral token.Literal
	}{
		{token.ASSIGN, token.Literal("=")},
		{token.PLUS, token.Literal("+")},
		{token.LPAREN, token.Literal("(")},
		{token.RPAREN, token.Literal(")")},
		{token.LBRACE, token.Literal("{")},
		{token.RBRACE, token.Literal("}")},
		{token.COMMA, token.Literal(",")},
		{token.SEMICOLON, token.Literal(";")},
		{token.EOF, token.Literal("")},
	}

	l := New(input)
	for i, tt := range tests {
		tok := l.NextToken()

		if tok.Type != tt.expectedType {
			t.Fatalf("tests[%d] - tokentype wrong. expected=%q, got=%q",
				i, tt.expectedType, tok.Type)
		}

		if tok.Literal != tt.expectedLiteral {
			t.Fatalf("tests[%d] - literal wrong. expected=%q, got=%q",
				i, tt.expectedLiteral, tok.Literal)
		}
	}
}
