"""
Unified LLM Client - Supports Ollama (local), DeepSeek, and Gemini with automatic fallback.
"""

import os
import re
import sys
import json
import requests as req
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding for unicode
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class LLMClient:
    """
    Unified interface for multiple LLM providers.
    Supports Ollama (local), DeepSeek, and Gemini with automatic fallback.

    Priority for research workloads:
    - Ollama: Best for bulk work (free, no rate limits, private)
    - DeepSeek: Great reasoning, good for synthesis
    - Gemini: Long context (1M tokens), fast for single queries
    """

    def __init__(self, primary: str = "ollama", fallback: str = "gemini",
                 ollama_model: str = None, ollama_url: str = None):
        """
        Initialize LLM client.

        Args:
            primary: Primary LLM ('ollama', 'deepseek', or 'gemini')
            fallback: Fallback LLM if primary fails
            ollama_model: Ollama model name (auto-detected if None)
            ollama_url: Ollama API URL (default: http://localhost:11434)
        """
        self.primary = primary
        self.fallback = fallback

        # Ollama setup (local, free, no rate limits)
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "")
        self.ollama_available = False
        self._init_ollama()

        # DeepSeek setup (OpenAI-compatible)
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_client = None
        if self.deepseek_api_key:
            try:
                from openai import OpenAI
                self.deepseek_client = OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                print("  ✓ DeepSeek client initialized")
            except ImportError:
                print("  ⚠️  OpenAI package not installed for DeepSeek")

        # Gemini setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("  ✓ Gemini client initialized")
            except ImportError:
                print("  ⚠️  google-generativeai package not installed")

        # Auto-fallback if primary not available
        if primary == "ollama" and not self.ollama_available:
            if self.deepseek_client:
                self.primary = "deepseek"
                print("  → Primary switched to DeepSeek (Ollama unavailable)")
            elif self.gemini_model:
                self.primary = "gemini"
                print("  → Primary switched to Gemini (Ollama unavailable)")

    def _init_ollama(self):
        """Detect Ollama and find the best available model."""
        try:
            resp = req.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                available_names = [m['name'] for m in models]

                if not available_names:
                    print("  ⚠️  Ollama running but no models installed")
                    return

                # If model specified, check it exists
                if self.ollama_model:
                    match = None
                    for name in available_names:
                        if name == self.ollama_model or name.startswith(self.ollama_model + ':'):
                            match = name
                            break
                    if match:
                        self.ollama_model = match
                        self.ollama_available = True
                        print(f"  ✓ Ollama: using {self.ollama_model}")
                        return
                    else:
                        print(f"  ⚠️  Ollama model '{self.ollama_model}' not found, auto-selecting...")

                # Auto-detect best model for research
                preferred = [
                    'deepseek-r1', 'qwen2.5', 'qwen3', 'llama3.1', 'llama3.3',
                    'gemma2', 'gemma3', 'phi4', 'mistral', 'command-r'
                ]

                for pref in preferred:
                    for name in available_names:
                        if name.startswith(pref):
                            self.ollama_model = name
                            self.ollama_available = True
                            print(f"  ✓ Ollama: auto-selected {self.ollama_model}")
                            return

                # Fallback to first available model
                self.ollama_model = available_names[0]
                self.ollama_available = True
                print(f"  ✓ Ollama: using {self.ollama_model}")

        except req.ConnectionError:
            print("  ⚠️  Ollama not running (start with: ollama serve)")
        except Exception as e:
            print(f"  ⚠️  Ollama detection failed: {e}")

    def get_available_providers(self) -> Dict[str, bool]:
        """Return which LLM providers are available."""
        return {
            'ollama': self.ollama_available,
            'deepseek': self.deepseek_client is not None,
            'gemini': self.gemini_model is not None
        }

    def get_ollama_models(self) -> List[str]:
        """List all available Ollama models."""
        try:
            resp = req.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return [m['name'] for m in resp.json().get('models', [])]
        except:
            pass
        return []

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict,
        temperature: float = 0.1,
        use_llm: Optional[str] = None
    ) -> Dict:
        """
        Generate structured JSON response from LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            response_schema: Expected JSON schema
            temperature: 0-1, lower = more deterministic
            use_llm: Force specific LLM ('ollama', 'deepseek', or 'gemini')

        Returns:
            Parsed JSON response
        """
        llm_to_use = use_llm or self.primary

        try:
            if llm_to_use == "ollama" and self.ollama_available:
                return self._call_ollama(system_prompt, user_prompt, response_schema, temperature)
            elif llm_to_use == "deepseek" and self.deepseek_client:
                return self._call_deepseek(system_prompt, user_prompt, response_schema, temperature)
            elif llm_to_use == "gemini" and self.gemini_model:
                return self._call_gemini(system_prompt, user_prompt, response_schema, temperature)
            else:
                raise ValueError(f"LLM '{llm_to_use}' not available")

        except Exception as e:
            print(f"  ⚠️  {llm_to_use} failed: {e}")

            # Try fallback chain: ollama -> deepseek -> gemini
            fallback_chain = ['ollama', 'deepseek', 'gemini']
            for fb in fallback_chain:
                if fb != llm_to_use and fb != use_llm:
                    avail = self.get_available_providers()
                    if avail.get(fb):
                        print(f"  🔄 Trying fallback: {fb}")
                        return self.generate_structured(
                            system_prompt, user_prompt, response_schema,
                            temperature, use_llm=fb
                        )
            raise

    def _call_ollama(self, system_prompt: str, user_prompt: str, schema: Dict, temp: float) -> Dict:
        """Call Ollama local API."""
        full_prompt = f"""{system_prompt}

User Query: {user_prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

IMPORTANT: Respond with ONLY valid JSON, no markdown code blocks, no explanation."""

        payload = {
            "model": self.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": 4096,
            },
            "format": "json"
        }

        resp = req.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text[:200]}")

        text = resp.json().get('response', '').strip()
        return self._extract_json(text)

    def _call_deepseek(self, system_prompt: str, user_prompt: str, schema: Dict, temp: float) -> Dict:
        """Call DeepSeek API"""
        response = self.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    def _call_gemini(self, system_prompt: str, user_prompt: str, schema: Dict, temp: float) -> Dict:
        """Call Gemini API"""
        import google.generativeai as genai

        full_prompt = f"""{system_prompt}

User Query: {user_prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

IMPORTANT: Respond with ONLY valid JSON, no markdown code blocks."""

        response = self.gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp,
            )
        )

        text = response.text.strip()
        return self._extract_json(text)

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response, handling markdown blocks and thinking tags."""
        # Strip <think>...</think> blocks (deepseek-r1 reasoning)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Handle markdown code blocks
        if '```json' in text:
            text = text.split('```json', 1)[1]
            text = text.split('```', 1)[0]
        elif '```' in text:
            text = text.split('```', 1)[1]
            text = text.split('```', 1)[0]

        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")

    def generate_text(self, prompt: str, temperature: float = 0.7,
                      use_llm: Optional[str] = None) -> str:
        """Generate plain text (non-structured) response."""
        llm_to_use = use_llm or self.primary

        if llm_to_use == "ollama" and self.ollama_available:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            resp = req.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            if resp.status_code == 200:
                text = resp.json().get('response', '')
                # Strip thinking tags for clean output
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                return text
            raise RuntimeError(f"Ollama error: {resp.status_code}")

        elif llm_to_use == "deepseek" and self.deepseek_client:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

        elif llm_to_use == "gemini" and self.gemini_model:
            response = self.gemini_model.generate_content(prompt)
            return response.text

        # Try fallback
        for fb in ['ollama', 'deepseek', 'gemini']:
            if fb != llm_to_use:
                avail = self.get_available_providers()
                if avail.get(fb):
                    return self.generate_text(prompt, temperature, use_llm=fb)

        raise ValueError("No LLM available")


def test_llm_client():
    """Test the LLM client"""
    client = LLMClient(primary="ollama", fallback="gemini")

    print("\nAvailable providers:", client.get_available_providers())

    if client.ollama_available:
        print(f"Ollama model: {client.ollama_model}")
        print(f"All Ollama models: {client.get_ollama_models()}")

    system_prompt = "You are a helpful assistant that analyzes academic research queries."
    user_prompt = "What is the research field for: 'Deep learning for protein structure prediction'?"

    schema = {
        "field": "string",
        "confidence": "float (0-1)",
        "reasoning": "string"
    }

    try:
        response = client.generate_structured(system_prompt, user_prompt, schema)
        print("✓ LLM Response:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_llm_client()
