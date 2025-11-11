"""Prompt templates for RAG system"""

from typing import List, Dict


SYSTEM_PROMPT = """You are an intelligent AI assistant for a WhatsApp chatbot. Your role is to provide helpful, accurate, and friendly responses based on the provided context.

Guidelines:
- Answer questions accurately using the context provided
- Be concise and clear - this is WhatsApp, keep responses brief
- Use a friendly, conversational tone with appropriate emojis
- If the context doesn't contain the answer, politely say you don't have that information
- Never make up information that's not in the context
- Format responses for readability (use line breaks, bullet points when needed)
- If relevant, provide actionable next steps
- Be respectful and professional at all times

Context Information:
{context}

Conversation History:
{conversation_history}

Now, answer the user's question based on the context above."""


FALLBACK_PROMPT = """You are a helpful AI assistant for a WhatsApp chatbot. The knowledge base doesn't contain specific information to answer this question, but you should provide a helpful response.

Guidelines:
- Acknowledge that you don't have specific information about their question
- Provide general helpful information if possible
- Suggest alternative ways they can get help
- Be friendly and apologetic
- Keep it brief and conversational

Conversation History:
{conversation_history}

User Question: {query}

Provide a helpful, friendly response."""


def build_system_message(context: str, conversation_history: str) -> str:
    """Build system message with context"""
    return SYSTEM_PROMPT.format(
        context=context,
        conversation_history=conversation_history
    )


def build_fallback_message(query: str, conversation_history: str) -> str:
    """Build fallback message when no context found"""
    return FALLBACK_PROMPT.format(
        query=query,
        conversation_history=conversation_history
    )


def format_conversation_history(messages: List[Dict[str, str]], max_messages: int = 5) -> str:
    """
    Format conversation history for prompt
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_messages: Maximum number of messages to include
        
    Returns:
        Formatted conversation history string
    """
    if not messages:
        return "No previous conversation."
    
    # Take last N messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted = []
    for msg in recent_messages:
        role = "User" if msg['role'] == 'user' else "Assistant"
        content = msg['content']
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)


def format_context(retrieved_docs: List[Dict]) -> str:
    """
    Format retrieved documents into context string
    
    Args:
        retrieved_docs: List of retrieved document dicts
        
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant information found in the knowledge base."
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        payload = doc.get('payload', {})
        title = payload.get('title', 'Document')
        content = payload.get('content', '')
        score = doc.get('score', 0)
        
        context_parts.append(
            f"--- Source {i}: {title} (Relevance: {score:.2f}) ---\n{content}"
        )
    
    return "\n\n".join(context_parts)


def build_messages(
    query: str,
    context: str,
    conversation_history: str
) -> List[Dict[str, str]]:
    """
    Build message list for LLM
    
    Args:
        query: User query
        context: Retrieved context
        conversation_history: Formatted conversation history
        
    Returns:
        List of message dicts for LLM API
    """
    system_message = build_system_message(context, conversation_history)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    return messages


def build_fallback_messages(
    query: str,
    conversation_history: str
) -> List[Dict[str, str]]:
    """
    Build fallback messages when no context found
    
    Args:
        query: User query
        conversation_history: Formatted conversation history
        
    Returns:
        List of message dicts for LLM API
    """
    system_message = build_fallback_message(query, conversation_history)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    return messages


# Quick reply examples (for future enhancement)
QUICK_REPLIES = {
    'hours': ['Business Hours', 'Contact Info', 'Location'],
    'products': ['View Products', 'Categories', 'Pricing'],
    'support': ['Talk to Human', 'FAQs', 'Submit Ticket'],
    'general': ['Help', 'About Us', 'Services']
}


def get_quick_replies(intent: str) -> List[str]:
    """Get quick reply suggestions based on intent"""
    return QUICK_REPLIES.get(intent, QUICK_REPLIES['general'])
