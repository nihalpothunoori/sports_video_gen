from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


SOCCERTRACK_V2_ACTIONS: List[str] = [
    "Pass",
    "Drive",
    "Shot",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
]


ACTION_ALIASES: Dict[str, List[str]] = {
    "Pass": ["pass", "through ball", "short pass", "one touch pass", "assist pass"],
    "Drive": ["drive", "dribble", "carry", "run with ball", "ball carry"],
    "Shot": ["shot", "shoot", "strike", "attempt on goal", "finish"],
    "Header": ["header", "headed", "head it"],
    "High Pass": ["high pass", "long ball", "lofted pass", "switch play"],
    "Out": ["out", "out of play", "goes wide", "ball out"],
    "Cross": ["cross", "crossing", "cutback", "whip it in"],
    "Throw In": ["throw in", "throw-in", "sideline throw"],
    "Ball Player Block": ["block", "blocked shot", "shot blocked", "deflection block"],
    "Player Successful Tackle": ["tackle", "successful tackle", "wins the ball", "clean tackle"],
    "Free Kick": ["free kick", "set piece", "fk", "direct free kick"],
    "Goal": ["goal", "scores", "finds the net", "puts it away"],
}


class SoccerTrackPromptMapper:
    # Maps free text prompts to SoccerTrack-style action vocabulary.
    def __init__(
        self,
        use_llm: bool = True,
        model: str = "gpt-4o-mini",
        api_base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> None:
        self.use_llm = use_llm
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, Tuple[str, List[str], str]] = {}

    def normalize_prompt(self, prompt: str) -> Tuple[str, List[str], str]:
        key = prompt.strip()
        if key in self._cache:
            return self._cache[key]

        actions: List[str]
        mapped_prompt: str
        source: str

        if self.use_llm and self.api_key:
            llm_out = self._map_with_llm(key)
            if llm_out is not None:
                mapped_prompt, actions = llm_out
                source = "llm"
            else:
                actions = self._map_with_rules(key)
                mapped_prompt = self._compose_prompt(key, actions)
                source = "rules_fallback"
        else:
            actions = self._map_with_rules(key)
            mapped_prompt = self._compose_prompt(key, actions)
            source = "rules"

        result = (mapped_prompt, actions, source)
        self._cache[key] = result
        return result

    def _map_with_rules(self, prompt: str) -> List[str]:
        text = f" {prompt.lower()} "
        found: List[str] = []
        for action in SOCCERTRACK_V2_ACTIONS:
            aliases = ACTION_ALIASES.get(action, [])
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, text):
                    found.append(action)
                    break
        return found

    def _map_with_llm(self, prompt: str) -> Optional[Tuple[str, List[str]]]:
        try:
            schema_hint = {
                "actions": ["Pass", "Shot"],
                "rewritten_prompt": "Broadcast view, fast Pass leading to a Shot near the box.",
            }
            system_text = (
                "You map soccer prompts to SoccerTrack v2 action vocabulary only. "
                f"Allowed actions: {', '.join(SOCCERTRACK_V2_ACTIONS)}. "
                "Return strict JSON with keys: actions (array of allowed action strings), "
                "rewritten_prompt (short improved prompt that naturally includes selected actions). "
                "No markdown and no extra text."
            )
            user_text = (
                f"Prompt: {prompt}\n"
                f"JSON example shape: {json.dumps(schema_hint)}"
            )
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url=f"{self.api_base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))

            content = response_json["choices"][0]["message"]["content"]
            parsed = self._parse_json_content(content)
            actions = self._sanitize_actions(parsed.get("actions", []))
            rewritten = str(parsed.get("rewritten_prompt", "")).strip()
            if not rewritten:
                rewritten = self._compose_prompt(prompt, actions)
            return rewritten, actions
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError):
            return None

    def _parse_json_content(self, content: str) -> Dict[str, object]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

    def _sanitize_actions(self, actions: List[object]) -> List[str]:
        allowed = set(SOCCERTRACK_V2_ACTIONS)
        clean: List[str] = []
        for item in actions:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            for action in SOCCERTRACK_V2_ACTIONS:
                if normalized.lower() == action.lower():
                    if action in allowed and action not in clean:
                        clean.append(action)
                    break
        return clean

    def _compose_prompt(self, prompt: str, actions: List[str]) -> str:
        if not actions:
            return prompt
        actions_str = ", ".join(actions)
        return f"{prompt.strip()} Soccer actions: {actions_str}."
