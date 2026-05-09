from openai import OpenAI
import os
import torch
from src.lapis.agents.agent import Agent
from typing import Optional

class GPTAgent(Agent):
    def __init__(self, model, max_history: int = 5, api_key: str | None = None):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # prompt_chain holds messages starting with the system prompt at index 0
        self.prompt_chain = []
        # max_history: number of user/assistant messages to keep (excluding the system prompt)
        self.max_history = None if max_history is None else int(max_history)
        
    def _chat(self, messages, temperature=None, top_p=None):
            completion_args = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "top_p": 1.0,
                "seed": 42  # Fixed seed for deterministic results
            }

            completion = self.client.chat.completions.create(**completion_args)
            return completion.choices[0].message.content

    def reset(self):
        self.prompt_chain = []
        torch.cuda.empty_cache()

    def llm_call(self, content: str, prompt: str, temperature=None, top_p=None):
        messages = [
            {"role": "system", "content": content},
            {"role": "user",   "content": prompt}
        ]
        return self._chat(messages, temperature=temperature, top_p=top_p)

    def init_prompt_chain(self, content: str, prompt: str):
        assert len(self.prompt_chain) == 0, "Prompt chain is not empty!"
        self.prompt_chain = [
            {"role": "system", "content": content},
            {"role": "user",   "content": prompt}
        ]
        self._enforce_history_limit()

    def update_prompt_chain(self, content: str, prompt: str):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        assert self.prompt_chain[0].get("role") == "system", "First message must be a system message."

        self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})
        self._enforce_history_limit()
        
    def update_prompt_chain_with_response(self, response: str, role: str = "assistant"):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        self.prompt_chain.append({"role": role, "content": response})
        self._enforce_history_limit()

    def query_msg_chain(self, temperature=None, top_p=None):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        return self._chat(self.prompt_chain, temperature=temperature, top_p=top_p)

    def _enforce_history_limit(self):
        """
        Ensure the prompt_chain does not exceed max_history user/assistant messages.
        Always keep the system prompt at index 0. When the limit is exceeded,
        remove the oldest message after the system prompt (index 1).
        """
        if self.max_history is None:
            return

        # Number of non-system messages currently stored
        non_system_count = max(0, len(self.prompt_chain) - 1)
        while non_system_count > self.max_history:
            # remove first message after the system prompt
            if len(self.prompt_chain) > 1:
                del self.prompt_chain[1]
            non_system_count = max(0, len(self.prompt_chain) - 1)


