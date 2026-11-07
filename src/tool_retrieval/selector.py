"""
Tool selection mechanisms: deterministic Cosine Selector and LLM Selector (Groq).
"""

from typing import List, Dict, Optional, Tuple, Any
import re
import time
import torch


class CosineSelector:
    """
    Deterministic Top-1 selector based on memory-penalized cosine similarity scores.
    """

    def select(self, scores: torch.Tensor, excluded_indices: Optional[set] = None) -> int:
        """
        Selects candidate index with highest score, respecting within-episode exclusions.
        """
        sc = scores.clone()
        if excluded_indices:
            for idx in excluded_indices:
                sc[idx] = -999.0
        return int(sc.argmax().item())


class GroqLLMSelector:
    """
    LLM-based selector using Groq API (e.g. openai/gpt-oss-20b) over a top-k candidate shortlist.
    Features response caching, exponential backoff, and robust integer parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-20b",
        temperature: float = 0.0,
        cache: Optional[Dict[Tuple[str, Tuple[str, ...]], int]] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.cache = cache if cache is not None else {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "groq package is required for GroqLLMSelector. Install with `pip install groq`."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def select_from_shortlist(
        self,
        query_text: str,
        shortlist_descriptions: List[str],
        max_retries: int = 5,
    ) -> Optional[int]:
        """
        Prompts LLM to select the most appropriate tool from the shortlist.

        Returns:
            Zero-based index of chosen tool within the shortlist, or None if failed.
        """
        cache_key = (query_text, tuple(shortlist_descriptions))
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"User Query: {query_text}\n\nAvailable Tools:\n"
        for i, desc in enumerate(shortlist_descriptions):
            prompt += f"{i + 1}. {desc}\n"
        prompt += "\nWhich tool should be used? Respond ONLY with the number of the tool."

        client = self._get_client()

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content or ""
                match = re.search(r"\b\d+\b", text)
                if match:
                    idx = int(match.group()) - 1
                    if 0 <= idx < len(shortlist_descriptions):
                        self.cache[cache_key] = idx
                        return idx
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "503" in err_str:
                    wait_time = 2 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    break

        return None
