"""Unified LLM Client supporting Google Gemini, OpenAI, and an offline Deterministic Engine.

Ensures zero dependency on paid APIs during automated evaluation or testing,
while seamlessly utilizing live Google Gemini or OpenAI when keys are present.
"""
import os
import json
from typing import List, Dict, Any, Optional
from src.config import GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER

# Check for google-genai availability
HAS_GEMINI = False
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    pass

# Check for openai availability
HAS_OPENAI = False
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.gemini_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        self.openai_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        self.client = None

        if self.provider == "gemini" and self.gemini_key and HAS_GEMINI:
            try:
                self.client = genai.Client(api_key=self.gemini_key)
                self.model = model or "gemini-2.5-flash"
            except Exception as e:
                print(f"Notice: Failed to initialize Gemini client ({e}). Defaulting to offline policy engine.")
                self.provider = "offline"
        elif self.provider == "openai" and self.openai_key and HAS_OPENAI:
            try:
                self.client = openai.OpenAI(api_key=self.openai_key)
                self.model = model or "gpt-4o-mini"
            except Exception as e:
                print(f"Notice: Failed to initialize OpenAI client ({e}). Defaulting to offline policy engine.")
                self.provider = "offline"
        else:
            self.provider = "offline"
            self.model = "deterministic-policy-engine"

    def generate(self, system_prompt: str, messages: List[Dict[str, str]],
                 context_chunks: Optional[List[str]] = None, temperature: float = 0.0) -> str:
        """Generate response from configured LLM or offline policy generator."""
        if self.provider == "gemini" and self.client:
            try:
                contents = []
                full_system = system_prompt
                if context_chunks:
                    full_system += "\n\nAUTHORITATIVE COMPANY EVIDENCE:\n" + "\n---\n".join(context_chunks)

                prompt_text = f"System: {full_system}\n"
                for m in messages:
                    prompt_text += f"{m['role'].capitalize()}: {m['content']}\n"
                prompt_text += "Assistant:"

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=300)
                )
                if resp.text:
                    return resp.text.strip()
            except Exception as e:
                print(f"Gemini API generation error: {e}. Falling back to offline engine.")

        elif self.provider == "openai" and self.client:
            try:
                api_messages = [{"role": "system", "content": system_prompt}]
                if context_chunks:
                    api_messages[0]["content"] += "\n\nAUTHORITATIVE COMPANY EVIDENCE:\n" + "\n---\n".join(context_chunks)
                for m in messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=300
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API generation error: {e}. Falling back to offline engine.")

        # Offline fallback: Handled by agent's deterministic policy engine
        return ""
