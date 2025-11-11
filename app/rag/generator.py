"""LLM response generator"""

from typing import List, Dict, Any, Optional, AsyncIterator
from openai import OpenAI, AsyncOpenAI
import tiktoken
import logging
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class Generator:
    """LLM-based response generator"""
    
    def __init__(self):
        self.client = OpenAI(api_key=rag_config.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=rag_config.openai_api_key)
        self.model = rag_config.llm_model
        self.max_tokens = rag_config.max_tokens
        self.temperature = rag_config.temperature
        
        # Token counter
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list"""
        total = 0
        for message in messages:
            # Each message has overhead (role, content, etc.)
            total += 4  # Overhead per message
            for key, value in message.items():
                total += self.count_tokens(str(value))
        total += 2  # Overhead for entire request
        return total
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (default: from config)
            max_tokens: Max tokens to generate (default: from config)
            
        Returns:
            Dict with 'text' and 'tokens' keys
        """
        try:
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens
            
            # Count input tokens
            input_tokens = self.count_messages_tokens(messages)
            logger.info(f"Generating response with {input_tokens} input tokens")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            text = response.choices[0].message.content
            
            # Token usage
            usage = response.usage
            total_tokens = usage.total_tokens if usage else input_tokens + self.count_tokens(text)
            
            logger.info(f"Generated response: {len(text)} chars, {total_tokens} total tokens")
            
            return {
                'text': text,
                'tokens': total_tokens,
                'input_tokens': usage.prompt_tokens if usage else input_tokens,
                'output_tokens': usage.completion_tokens if usage else self.count_tokens(text)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM (async)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Dict with 'text' and 'tokens' keys
        """
        try:
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens
            
            # Count input tokens
            input_tokens = self.count_messages_tokens(messages)
            logger.info(f"Generating response (async) with {input_tokens} input tokens")
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            text = response.choices[0].message.content
            
            # Token usage
            usage = response.usage
            total_tokens = usage.total_tokens if usage else input_tokens + self.count_tokens(text)
            
            logger.info(f"Generated response (async): {len(text)} chars, {total_tokens} total tokens")
            
            return {
                'text': text,
                'tokens': total_tokens,
                'input_tokens': usage.prompt_tokens if usage else input_tokens,
                'output_tokens': usage.completion_tokens if usage else self.count_tokens(text)
            }
            
        except Exception as e:
            logger.error(f"Error generating response (async): {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate response with streaming (async)
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Yields:
            Response chunks as they arrive
        """
        try:
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens
            
            logger.info(f"Starting streaming response generation")
            
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            logger.info(f"Streaming response completed")
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def format_for_whatsapp(self, text: str, max_length: int = 4000) -> List[str]:
        """
        Format response for WhatsApp (split if too long)
        
        Args:
            text: Response text
            max_length: Max length per message (WhatsApp limit ~4096)
            
        Returns:
            List of message chunks
        """
        if len(text) <= max_length:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split response into {len(chunks)} chunks for WhatsApp")
        return chunks


# Global generator instance
generator = Generator()
