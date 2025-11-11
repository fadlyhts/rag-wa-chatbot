"""Google Gemini LLM generator"""

from typing import List, Dict, Any, Optional, AsyncIterator
import google.generativeai as genai
import logging
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """Gemini-based response generator"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=rag_config.google_api_key)
        self.model_name = rag_config.gemini_model
        self.max_tokens = rag_config.gemini_max_tokens
        self.temperature = rag_config.gemini_temperature
        
        # Initialize model
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Convert OpenAI-style messages to Gemini format
        
        Gemini uses a different format:
        - System message goes in the model config
        - User/assistant messages become the chat history
        """
        system_instruction = None
        chat_history = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                chat_history.append({"role": "model", "parts": [content]})
        
        return system_instruction, chat_history
    
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
            # Convert messages to Gemini format
            system_instruction, chat_history = self._convert_messages_to_gemini_format(messages)
            
            # Update generation config if overrides provided
            generation_config = genai.GenerationConfig(
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content")
                    break
            
            if not user_message:
                raise ValueError("No user message found in messages")
            
            # Generate response
            if len(chat_history) > 1:
                # Use chat if there's history
                chat = model.start_chat(history=chat_history[:-1])
                response = chat.send_message(user_message)
            else:
                # Direct generation for single message
                response = model.generate_content(user_message)
            
            content = response.text
            
            # Count tokens
            prompt_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in messages)
            completion_tokens = self.count_tokens(content)
            total_tokens = prompt_tokens + completion_tokens
            
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
        """
        Generate response asynchronously
        
        Note: Gemini SDK doesn't have native async support yet,
        so we wrap the sync call
        """
        return self.generate(messages, temperature, max_tokens, **kwargs)
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate response as a stream
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Yields:
            Text chunks as they are generated
        """
        try:
            # Convert messages to Gemini format
            system_instruction, chat_history = self._convert_messages_to_gemini_format(messages)
            
            # Update generation config
            generation_config = genai.GenerationConfig(
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            
            # Create model
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content")
                    break
            
            if not user_message:
                raise ValueError("No user message found in messages")
            
            # Generate streaming response
            if len(chat_history) > 1:
                chat = model.start_chat(history=chat_history[:-1])
                response = chat.send_message(user_message, stream=True)
            else:
                response = model.generate_content(user_message, stream=True)
            
            # Stream chunks
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise


# Global Gemini generator instance
gemini_generator = GeminiGenerator()
