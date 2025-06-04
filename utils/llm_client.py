"""
LLM client for integrating with various language models.
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from config import LLM_CONFIG, MOCK_LLM

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from the LLM."""
        pass
    
    @abstractmethod
    def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Classify text into one of the given categories."""
        pass


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without actual LLM."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        if "classify" in prompt.lower():
            return "email|normal|Customer inquiry about product pricing"
        elif "extract" in prompt.lower():
            return "sender: john.doe@email.com, subject: Product Inquiry, urgency: normal"
        else:
            return "Mock LLM response for testing purposes."
    
    def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Return mock classification results."""
        return {
            "category": categories[0] if categories else "Other",
            "confidence": 0.85,
            "reasoning": "Mock classification for testing"
        }


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM server."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or LLM_CONFIG
        self.api_base = self.config["api_base"]
        self.model_name = self.config["model_name"]
        self.timeout = self.config["timeout"]
    
    def _make_request(self, prompt: str, **kwargs) -> str:
        """Make a request to the Ollama API and return a decoded string."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config["temperature"]),
                    "num_predict": kwargs.get("max_tokens", self.config["max_tokens"]),
                }
            }
            
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get("response", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            return text.strip()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from Ollama."""
        try:
            response = self._make_request(prompt, **kwargs)
            if isinstance(response, bytes):
                return response.decode("utf-8").strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        except Exception as e:
            logger.error(f"Ollama generate() failed: {e}")
            return "unknown,0.3"  # Default fallback format

    
    def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Classify text using Ollama."""
        categories_str = ", ".join(categories)
        prompt = f"""
        Classify the following text into one of these categories: {categories_str}
        
        Text: {text}
        
        Respond with a JSON object containing:
        - category: the most appropriate category
        - confidence: confidence score between 0 and 1
        - reasoning: brief explanation for the classification
        
        Response:
        """
        
        try:
            response = self._make_request(prompt)
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3].strip()
            elif response.startswith("```"):
                response = response[3:-3].strip()
            
            result = json.loads(response)
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")
            # Fallback: simple keyword matching
            text_lower = text.lower()
            for category in categories:
                if category.lower() in text_lower:
                    return {
                        "category": category,
                        "confidence": 0.6,
                        "reasoning": f"Keyword match for '{category}'"
                    }
            
            return {
                "category": categories[0] if categories else "Other",
                "confidence": 0.3,
                "reasoning": "Fallback classification due to parsing error"
            }


class LangChainLLMClient(BaseLLMClient):
    """Client using LangChain for LLM integration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or LLM_CONFIG
        self._init_langchain()
    
    def _init_langchain(self):
        """Initialize LangChain LLM."""
        try:
            from langchain.llms import Ollama
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            self.llm = Ollama(
                model=self.config["model_name"],
                base_url=self.config["api_base"],
                temperature=self.config["temperature"],
            )
            
            self.classification_template = PromptTemplate(
                input_variables=["text", "categories"],
                template="""
                Classify the following text into one of these categories: {categories}
                
                Text: {text}
                
                Respond with only the category name that best fits.
                """
            )
            
            self.classification_chain = LLMChain(
                llm=self.llm,
                prompt=self.classification_template
            )
            
        except ImportError as e:
            logger.error(f"LangChain not available: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LangChain."""
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            raise
    
    def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Classify text using LangChain."""
        try:
            categories_str = ", ".join(categories)
            result = self.classification_chain.run(
                text=text,
                categories=categories_str
            )
            
            # Find the matching category
            result_lower = result.lower().strip()
            matched_category = None
            
            for category in categories:
                if category.lower() in result_lower:
                    matched_category = category
                    break
            
            return {
                "category": matched_category or categories[0],
                "confidence": 0.8,
                "reasoning": f"LangChain classification: {result}"
            }
            
        except Exception as e:
            logger.error(f"LangChain classification failed: {e}")
            return {
                "category": categories[0] if categories else "Other",
                "confidence": 0.3,
                "reasoning": f"Classification failed: {str(e)}"
            }


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (supports various models including Llama)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or LLM_CONFIG
        self.api_key = self.config["api_key"]
        self.model_name = self.config["model_name"]
        self.api_base = "https://openrouter.ai/api/v1"
        self.timeout = self.config["timeout"]
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
    
    def _make_chat_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a chat completion request to OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://multi-agent-system.local",
                "X-Title": "Multi-Agent AI System",
            }
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config["temperature"]),
                "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected OpenRouter API response format: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter request: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from OpenRouter."""
        messages = [{"role": "user", "content": prompt}]
        return self._make_chat_request(messages, **kwargs)
    
    def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Classify text using OpenRouter."""
        categories_str = ", ".join(categories)
        prompt = f"""
        Classify the following text into one of these categories: {categories_str}
        
        Text: "{text}"
        
        Respond with a JSON object containing:
        - category: the most appropriate category from the list
        - confidence: confidence score between 0 and 1
        - reasoning: brief explanation for the classification
        
        Example response:
        {{"category": "Invoice", "confidence": 0.9, "reasoning": "Contains invoice number and payment terms"}}
        """
        
        try:
            response = self.generate(prompt)
            
            # Try to extract JSON from response
            response = response.strip()
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif response.startswith("{") and response.endswith("}"):
                pass  # Already JSON format
            else:
                # Try to find JSON-like content
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    response = response[start:end]
            
            result = json.loads(response)
            
            # Validate the category is in our list
            if result.get("category") not in categories:
                # Find closest match
                text_lower = text.lower()
                for category in categories:
                    if category.lower() in text_lower:
                        result["category"] = category
                        result["confidence"] = max(0.3, result.get("confidence", 0.5) - 0.1)
                        result["reasoning"] += f" (Corrected to valid category: {category})"
                        break
                else:
                    result["category"] = categories[0]
                    result["confidence"] = 0.3
                    result["reasoning"] = "Fallback to first category due to invalid classification"
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")
            # Fallback: simple keyword matching
            text_lower = text.lower()
            for category in categories:
                if category.lower() in text_lower:
                    return {
                        "category": category,
                        "confidence": 0.6,
                        "reasoning": f"Keyword match for '{category}' (LLM parsing failed)"
                    }
            
            return {
                "category": categories[0] if categories else "Other",
                "confidence": 0.3,
                "reasoning": "Fallback classification due to parsing error"
            }


def get_llm_client() -> BaseLLMClient:
    """Factory function to get the appropriate LLM client."""
    if MOCK_LLM:
        logger.info("Using Mock LLM Client")
        return MockLLMClient()
    
    # Check if we have OpenRouter configuration
    if LLM_CONFIG.get("api_key") and "openrouter" in LLM_CONFIG.get("api_base", "").lower():
        logger.info("Using OpenRouter LLM Client")
        return OpenRouterClient()
    
    # Check if we have Ollama configuration
    elif "ollama" in LLM_CONFIG.get("api_base", "").lower() or LLM_CONFIG.get("api_base", "").startswith("http://localhost"):
        logger.info("Using Ollama LLM Client")
        return OllamaClient()
    
    # Try LangChain as fallback
    try:
        logger.info("Using LangChain LLM Client")
        return LangChainLLMClient()
    except ImportError:
        logger.warning("LangChain not available, falling back to Mock LLM")
        return MockLLMClient()


# Global LLM client instance
llm_client = get_llm_client()