import os
from anthropic import Anthropic
from src.lapis.agents.agent import Agent


class ClaudeAgent(Agent):
    def __init__(self, model, max_history: int = 5, api_key: str | None = None):
        super().__init__(model)
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = None
        use_openrouter_fallback = os.getenv("LAPIS_USE_OPENROUTER_FALLBACK", "false").lower() in {
            "1", "true", "yes", "on"
        }

        if not api_key and use_openrouter_fallback and os.getenv("OPENROUTER_API_KEY"):
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"

        if not api_key:
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY. Direct Anthropic mode is active. "
                "Set ANTHROPIC_API_KEY, or explicitly enable OPENROUTER fallback with "
                "LAPIS_USE_OPENROUTER_FALLBACK=true and OPENROUTER_API_KEY."
            )
        
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.prompt_chain = []
        self.max_history = None if max_history is None else int(max_history)

    def _chat(self, messages, temperature=None, top_p=None):
        # Separate system message from user/assistant turns
        system_prompt = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                conv_messages.append(m)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": conv_messages if conv_messages else [{"role": "user", "content": "Hi"}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def llm_call(self, prompt, question, **kwargs):
        messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": question})
        return self._chat(messages)

    def reset(self):
        self.prompt_chain = []
