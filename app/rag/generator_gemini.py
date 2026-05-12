"""Google Gemini LLM generator (using google-genai SDK)"""

from typing import List, Dict, Any, Optional, AsyncIterator
from google import genai
from google.genai import types
import logging
from app.rag.config import rag_config
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """Gemini-based response generator (new google-genai SDK)"""
    
    def __init__(self):
        # Initialize the new genai client
        self.client = genai.Client(api_key=rag_config.google_api_key)
        
        # Model name (without "models/" prefix)
        model_name = rag_config.gemini_model
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")
        self.model_name = model_name
        
        self.max_tokens = rag_config.gemini_max_tokens
        self.temperature = rag_config.gemini_temperature
        
        logger.info(f"Initializing Gemini generator with model: {self.model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=text,
            )
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text) // 4
    
    def _build_contents_and_config(self, messages: List[Dict[str, str]], temperature: Optional[float], max_tokens: Optional[int]):
        """
        Convert OpenAI-style messages to google-genai format.
        
        Returns:
            tuple: (contents, config)
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        
        return contents, config
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Gemini
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with 'content', 'tokens_used', and 'model' keys
        """
        try:
            contents, config = self._build_contents_and_config(messages, temperature, max_tokens)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            
            content = response.text
            
            # Get token usage from response metadata
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                total_tokens = response.usage_metadata.total_token_count or 0
            
            logger.info(f"Gemini generation: {total_tokens} tokens used")
            
            return {
                "content": content,
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response asynchronously"""
        try:
            contents, config = self._build_contents_and_config(messages, temperature, max_tokens)
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            
            content = response.text
            
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                total_tokens = response.usage_metadata.total_token_count or 0
            
            return {
                "content": content,
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Gemini async generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate response as a stream"""
        try:
            contents, config = self._build_contents_and_config(messages, temperature, max_tokens)
            
            async for chunk in self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
            
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def format_for_whatsapp(self, text: str, max_length: int = 4096) -> List[str]:
        """
        Format response text for WhatsApp.
        Splits long messages into multiple parts if needed.
        """
        if len(text) <= max_length:
            return [text]
        
        messages = []
        current_message = ""
        
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_message) + len(para) + 2 <= max_length:
                current_message += para + '\n\n'
            else:
                if current_message:
                    messages.append(current_message.strip())
                    current_message = para + '\n\n'
                else:
                    sentences = para.split('. ')
                    for sentence in sentences:
                        if len(current_message) + len(sentence) + 2 <= max_length:
                            current_message += sentence + '. '
                        else:
                            if current_message:
                                messages.append(current_message.strip())
                            current_message = sentence + '. '
        
        if current_message:
            messages.append(current_message.strip())
        
        return messages if messages else [text[:max_length]]


# Global Gemini generator instance
gemini_generator = GeminiGenerator()


class GeminiLCELWrapper:
    """
    Wraps GeminiGenerator into an LCEL-compatible callable
    (accepts LangChain message list, returns string).
    """

    def invoke(self, messages: list) -> str:
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})

        result = gemini_generator.generate(converted)
        return result.get("content") or result.get("text", "")

    async def ainvoke(self, messages: list) -> str:
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})

        result = await gemini_generator.generate_async(converted)
        return result.get("content") or result.get("text", "")
