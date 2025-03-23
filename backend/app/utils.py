from contextlib import contextmanager
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Track token usage and costs for API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    cost: float = 0.0
    estimated_tokens: int = 0
    max_tokens: int = 8000  # Default max tokens for GPT-4

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def can_fit_in_context(self, text: str) -> bool:
        """Check if text can fit within context window.
        
        Args:
            text: The text to check
            
        Returns:
            True if text can fit, False otherwise
        """
        estimated = self.estimate_tokens(text)
        return estimated <= self.max_tokens

    def optimize_context(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Optimize context to fit within token limit.
        
        Args:
            text: The text to optimize
            max_tokens: Optional maximum tokens to fit within
            
        Returns:
            Optimized text that fits within token limit
        """
        max_tokens = max_tokens or self.max_tokens
        estimated = self.estimate_tokens(text)
        
        if estimated <= max_tokens:
            return text
            
        # If text is too long, truncate intelligently
        # Keep the most recent/relevant parts
        words = text.split()
        current_tokens = 0
        optimized_words = []
        
        for word in reversed(words):
            word_tokens = self.estimate_tokens(word)
            if current_tokens + word_tokens > max_tokens:
                break
            optimized_words.insert(0, word)
            current_tokens += word_tokens
            
        return " ".join(optimized_words)

    def update(self, response: Dict[str, Any]) -> None:
        """Update token usage from API response.
        
        Args:
            response: API response containing usage information
            
        Raises:
            ValueError: If response is missing required fields
        """
        try:
            usage = response.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            
            # Calculate cost based on OpenAI's pricing
            # GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
            self.cost = (self.prompt_tokens / 1000) * 0.03 + (self.completion_tokens / 1000) * 0.06
        except Exception as e:
            logger.error(f"Error updating token usage: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def finish(self) -> None:
        """Mark the end of token usage tracking."""
        self.end_time = time.time()

    def __str__(self) -> str:
        """String representation of token usage.
        
        Returns:
            Formatted string with usage statistics
        """
        duration = round(self.end_time - self.start_time, 2) if self.end_time else 0
        return (
            f"Tokens: {self.total_tokens} "
            f"(Prompt: {self.prompt_tokens}, "
            f"Completion: {self.completion_tokens}), "
            f"Cost: ${self.cost:.6f}, "
            f"Duration: {duration}s"
        )

@contextmanager
def track_token_usage():
    """Context manager for tracking token usage in API calls.
    
    Yields:
        TokenUsage instance for tracking usage
        
    Example:
        >>> with track_token_usage() as usage:
        ...     response = api_call()
        ...     usage.update(response)
    """
    token_usage = TokenUsage()
    token_usage.start_time = time.time()
    
    try:
        yield token_usage
    except Exception as e:
        logger.error(f"Error in token usage tracking: {str(e)}")
        raise
    finally:
        token_usage.finish()
        logger.info(token_usage)

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime object to ISO format string.
    
    Args:
        dt: Datetime object to format, defaults to current time
        
    Returns:
        ISO formatted timestamp string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()

def safe_json_loads(s: str) -> Any:
    """Safely load JSON string.
    
    Args:
        s: JSON string to parse
        
    Returns:
        Parsed JSON data
        
    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValueError(f"Invalid JSON: {str(e)}")