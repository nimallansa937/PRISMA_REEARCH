"""
Unified LLM Client - Supports DeepSeek and Gemini with automatic fallback.
"""

import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified interface for multiple LLM providers.
    Supports DeepSeek and Gemini with automatic fallback.
    """
    
    def __init__(self, primary: str = "deepseek", fallback: str = "gemini"):
        """
        Initialize LLM client.
        
        Args:
            primary: Primary LLM to use ('deepseek' or 'gemini')
            fallback: Fallback LLM if primary fails
        """
        self.primary = primary
        self.fallback = fallback
        
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
                print("✓ DeepSeek client initialized")
            except ImportError:
                print("⚠️  OpenAI package not installed for DeepSeek")
        
        # Gemini setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✓ Gemini client initialized")
            except ImportError:
                print("⚠️  google-generativeai package not installed")
    
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
            use_llm: Force specific LLM ('deepseek' or 'gemini')
        
        Returns:
            Parsed JSON response
        """
        llm_to_use = use_llm or self.primary
        
        try:
            if llm_to_use == "deepseek" and self.deepseek_client:
                return self._call_deepseek(system_prompt, user_prompt, response_schema, temperature)
            elif llm_to_use == "gemini" and self.gemini_model:
                return self._call_gemini(system_prompt, user_prompt, response_schema, temperature)
            else:
                raise ValueError(f"LLM '{llm_to_use}' not available")
                
        except Exception as e:
            print(f"⚠️  {llm_to_use} failed: {e}")
            
            # Try fallback
            if self.fallback and llm_to_use != self.fallback:
                print(f"🔄 Trying fallback: {self.fallback}")
                return self.generate_structured(
                    system_prompt, user_prompt, response_schema, temperature, use_llm=self.fallback
                )
            else:
                raise
    
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
        
        # Extract JSON from response
        text = response.text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
    
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate plain text (non-structured) response"""
        if self.primary == "deepseek" and self.deepseek_client:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        elif self.gemini_model:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        else:
            raise ValueError("No LLM available")


def test_llm_client():
    """Test the LLM client"""
    client = LLMClient(primary="gemini", fallback="deepseek")
    
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
