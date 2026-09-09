"""
Model Adapter Interface for QueryBot
Supports multiple LLM providers through a unified interface
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from loguru import logger
import httpx


class ModelAdapter(ABC):
    """Abstract base class for LLM model adapters"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from messages"""
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response from messages"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class OpenRouterAdapter(ModelAdapter):
    """
    Adapter for OpenRouter API - provides access to 100+ models
    https://openrouter.ai/
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str, default_model: str = "meta-llama/llama-3-70b-instruct"):
        self.api_key = api_key
        self.default_model = default_model
        self.client = httpx.Client(timeout=60.0)
        self._supported_models = None
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate completion using OpenRouter
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID (optional, uses default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
        """
        model = model or self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://querybot.ipac.ai",
            "X-Title": "QueryBot Political Intelligence"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        
        try:
            response = self.client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Log usage for observability
            if "usage" in result:
                logger.info(f"OpenRouter usage: {result['usage']}")
            
            return content
            
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter generation: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream completion using OpenRouter"""
        model = model or self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://querybot.ipac.ai"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            with self.client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        if self._supported_models is None:
            try:
                response = self.client.get(f"{self.BASE_URL}/models")
                if response.status_code == 200:
                    self._supported_models = response.json().get("data", [])
            except Exception as e:
                logger.error(f"Failed to fetch model info: {e}")
                self._supported_models = []
        
        return {
            "provider": "OpenRouter",
            "default_model": self.default_model,
            "supported_models_count": len(self._supported_models),
            "models": self._supported_models[:20]  # Return first 20 for brevity
        }
    
    def list_models(self) -> List[str]:
        """List available model IDs"""
        info = self.get_model_info()
        return [m.get("id", "") for m in info.get("models", [])]


class LocalModelAdapter(ModelAdapter):
    """Adapter for local models (Ollama, vLLM, etc.)"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=120.0)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using local model"""
        # Convert to Ollama format
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json().get("response", "")
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream from local model"""
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts) + "\nAssistant: "
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model info"""
        return {
            "provider": "Local",
            "base_url": self.base_url,
            "model": self.model
        }


class ModelAdapterFactory:
    """Factory for creating model adapters"""
    
    _adapters: Dict[str, ModelAdapter] = {}
    
    @classmethod
    def register(cls, name: str, adapter: ModelAdapter):
        """Register a model adapter"""
        cls._adapters[name] = adapter
        logger.info(f"Registered model adapter: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ModelAdapter]:
        """Get adapter by name"""
        return cls._adapters.get(name)
    
    @classmethod
    def create_openrouter(cls, api_key: str, model: str = None) -> ModelAdapter:
        """Create OpenRouter adapter"""
        adapter = OpenRouterAdapter(api_key=api_key, default_model=model)
        cls.register("openrouter", adapter)
        return adapter
    
    @classmethod
    def create_local(cls, base_url: str = None, model: str = None) -> ModelAdapter:
        """Create local model adapter"""
        adapter = LocalModelAdapter(base_url=base_url, model=model)
        cls.register("local", adapter)
        return adapter
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List registered adapter names"""
        return list(cls._adapters.keys())


def get_model_adapter(provider: str = "openrouter", **kwargs) -> ModelAdapter:
    """
    Get or create a model adapter
    
    Args:
        provider: Provider name ('openrouter', 'local')
        **kwargs: Provider-specific arguments
    
    Returns:
        ModelAdapter instance
    """
    from config.settings import settings
    
    if provider == "openrouter":
        api_key = kwargs.get("api_key") or getattr(settings, 'OPENROUTER_API_KEY', None)
        model = kwargs.get("model") or getattr(settings, 'LLM_MODEL', 'meta-llama/llama-3-70b-instruct')
        return ModelAdapterFactory.create_openrouter(api_key=api_key, model=model)
    
    elif provider == "local":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        model = kwargs.get("model", "llama3")
        return ModelAdapterFactory.create_local(base_url=base_url, model=model)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
