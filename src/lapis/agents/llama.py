# llama.py
import json
import os
import requests
from typing import List, Dict, Optional
from src.lapis.agents.agent import Agent

class LlamaAgent(Agent):
    """
    Llama-based agent with the SAME public API as GPTAgent:
      - _chat(messages, temperature=None, top_p=None)
      - reset()
      - llm_call(content, prompt, temperature=None, top_p=None)
      - init_prompt_chain(content, prompt)
      - update_prompt_chain(content, prompt)
      - update_prompt_chain_with_response(response, role="assistant")
      - query_msg_chain(temperature=None, top_p=None)
    """
    def __init__(self, model: str, base_url: Optional[str] = None, timeout: float = 120.0):
        super().__init__(model)
        self.base_url = base_url or os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout = timeout
        self.prompt_chain: List[Dict[str, str]] = []

    # --- Internal chat helper mirroring GPTAgent._chat ---
    def _chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        """
        Send a chat-style request to an Ollama-compatible server.

        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        """
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        # Ollama advanced parameters go under "options"
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if options:
            payload["options"] = options

        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # Expected chat response shape: {"message": {"role": "assistant", "content": "..."} , ...}
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            # Fallback for rare implementations that return "messages" list
            messages_out = data.get("messages", [])
            if messages_out and isinstance(messages_out[-1], dict):
                content = messages_out[-1].get("content", "")
        return content or ""

    def reset(self):
        self.prompt_chain = []
        # Mirror GPTAgent behavior while being safe if torch isn't installed
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def llm_call(self, content: str, prompt: str, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
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

    def update_prompt_chain(self, content: str, prompt: str):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        assert self.prompt_chain[0].get("role") == "system", "First message must be a system message."
        self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})

    def update_prompt_chain_with_response(self, response: str, role: str = "assistant"):
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        self.prompt_chain.append({"role": role, "content": response})

    def query_msg_chain(self, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        assert self.prompt_chain, "Prompt chain is empty. Call init_prompt_chain first."
        return self._chat(self.prompt_chain, temperature=temperature, top_p=top_p)
