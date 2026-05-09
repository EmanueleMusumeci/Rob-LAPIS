from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, model):
        self.model = model
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.tokens_thinking: int = 0
        self.calls: int = 0

    def reset_token_counts(self):
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_thinking = 0
        self.calls = 0

    def token_stats(self) -> dict:
        return {
            "llm_calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_thinking": self.tokens_thinking,
            "tokens_total": self.tokens_in + self.tokens_out + self.tokens_thinking,
        }

    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def llm_call(self, prompt, question, **kwargs) -> str:
        pass