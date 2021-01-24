# Internal
from ...parsing.LexerBase import LexerBase
from ...parsing.Token import Token
from .TokenKindNLPresentationFormat import TokenKindNLPresentationFormat

class LexerNLPresentationFormat(LexerBase[TokenKindNLPresentationFormat]):
    def on_lex_next(self) -> Token[TokenKindNLPresentationFormat]:
        if self.get_next_char() == "{":
            self.eat_next_char()
            if self.get_next_char() == "{":
                self.eat_next_char()
                while self.are_more_chars() and self.get_next_char() != "{":
                    self.eat_next_char()
                return self.pop_token(TokenKindNLPresentationFormat.TEXT)
            else:
                while self.are_more_chars() and self.get_next_char() != "}":
                    self.eat_next_char()
                if self.get_next_char() == "}":
                    self.eat_next_char()
                return self.pop_token(TokenKindNLPresentationFormat.FIELD_PATH)
        else:
            while self.are_more_chars() and self.get_next_char() != "{":
                self.eat_next_char()
            return self.pop_token(TokenKindNLPresentationFormat.TEXT)
