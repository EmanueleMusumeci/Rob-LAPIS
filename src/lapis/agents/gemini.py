import os
from google import genai
from google.genai import types
from src.lapis.agents.agent import Agent


class GeminiAgent(Agent):
    def __init__(self, model, max_history: int = 5, api_key: str | None = None):
        super().__init__(model)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it as an environment variable.")
        self.client = genai.Client(api_key=api_key)
        self.prompt_chain = []
        self.max_history = None if max_history is None else int(max_history)

    def _chat(self, messages, temperature=None, top_p=None):
        system_prompt = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))

        _thinking_models = ("flash-thinking", "gemini-2.5", "gemini-exp")
        thinking_cfg = (
            types.ThinkingConfig(thinking_budget=1024)
            if any(s in self.model for s in _thinking_models)
            else None
        )
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.0,
            max_output_tokens=32768,
            thinking_config=thinking_cfg,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents or [types.Content(role="user", parts=[types.Part(text="Hi")])],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.tokens_in += getattr(usage, "prompt_token_count", 0) or 0
            self.tokens_out += getattr(usage, "candidates_token_count", 0) or 0
            self.tokens_thinking += getattr(usage, "thoughts_token_count", 0) or 0
        self.calls += 1
        # Log non-STOP finish reasons so truncation is visible in logs
        try:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                import logging as _logging
                _logging.getLogger("my_logger").warning(
                    "[GEMINI] finish_reason=%s (tokens_out=%s) — response may be truncated",
                    finish_reason,
                    getattr(usage, "candidates_token_count", "?") if usage else "?",
                )
        except Exception:
            pass
        return response.text

    def llm_call(self, prompt, question, temperature=None, **kwargs):
        messages = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": question})
        return self._chat(messages, temperature=temperature)

    def reset(self):
        self.prompt_chain = []

    def init_prompt_chain(self, content: str, prompt: str):
        assert len(self.prompt_chain) == 0, "Prompt chain is not empty!"
        self.prompt_chain = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]
        self._enforce_history_limit()

    def update_prompt_chain(self, content: str, prompt: str):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})
        self._enforce_history_limit()

    def update_prompt_chain_with_response(self, response: str, role: str = "assistant"):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        self.prompt_chain.append({"role": role, "content": response})
        self._enforce_history_limit()

    def query_msg_chain(self, temperature=None, top_p=None):
        return self._chat(self.prompt_chain, temperature=temperature, top_p=top_p)

    def _enforce_history_limit(self):
        if self.max_history is None:
            return
        system = [m for m in self.prompt_chain if m["role"] == "system"]
        conv = [m for m in self.prompt_chain if m["role"] != "system"]
        if len(conv) > self.max_history * 2:
            conv = conv[-(self.max_history * 2):]
        self.prompt_chain = system + conv
