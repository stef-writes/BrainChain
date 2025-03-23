from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    STABILITY = "stability-ai"

class ModelType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AGENT = "agent"

class TaskType(str, Enum):
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"

class BaseModelConfig(BaseModel):
    """Base configuration class for all model types."""
    model_provider: ModelProvider = Field(default=ModelProvider.OPENAI)
    model_type: ModelType = Field(default=ModelType.TEXT)
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    task_type: TaskType = Field(default=TaskType.CONVERSATIONAL)

    @validator('api_key_env_var')
    def validate_api_key_env_var(cls, v):
        if not v:
            raise ValueError("api_key_env_var cannot be empty")
        return v

class LLMConfig(BaseModelConfig):
    """Configuration for language models."""
    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    fallback_models: List[str] = Field(default_factory=lambda: ["gpt-3.5-turbo"])
    adaptive_temperature: bool = Field(default=True)
    adaptive_max_tokens: bool = Field(default=True)

    @validator('model')
    def validate_model(cls, v):
        if not v:
            raise ValueError("model name cannot be empty")
        if v.startswith("claude"):
            return ModelProvider.ANTHROPIC
        return v

    @validator('model_provider')
    def set_provider(cls, v, values):
        if 'model' in values and values['model'].startswith("claude"):
            return ModelProvider.ANTHROPIC
        return v

    def adapt_parameters(self, input_length: int, task_type: Optional[TaskType] = None) -> None:
        """Adapt model parameters based on input length and task type.
        
        Args:
            input_length: Length of the input text
            task_type: Optional task type to adapt parameters for
        """
        if task_type:
            self.task_type = task_type
            
        # Adapt temperature based on task type
        if self.adaptive_temperature:
            if self.task_type == TaskType.FACTUAL:
                self.temperature = min(0.3, self.temperature)
            elif self.task_type == TaskType.CREATIVE:
                self.temperature = max(0.7, self.temperature)
            elif self.task_type == TaskType.ANALYTICAL:
                self.temperature = 0.5
                
        # Adapt max_tokens based on input length
        if self.adaptive_max_tokens:
            # Reserve some tokens for the response
            available_tokens = 8000 - input_length  # Assuming 8K context window
            self.max_tokens = min(
                max(100, available_tokens // 2),  # Use at least 100 tokens, at most half of available
                self.max_tokens
            )

    def get_fallback_model(self) -> Optional[str]:
        """Get the next available fallback model.
        
        Returns:
            The next available fallback model or None if none available
        """
        for model in self.fallback_models:
            if model.startswith("gpt-") and os.getenv("OPENAI_API_KEY"):
                return model
            elif model.startswith("claude-") and os.getenv("ANTHROPIC_API_KEY"):
                return model
        return None

    def should_fallback(self, error: Exception) -> bool:
        """Determine if we should try a fallback model.
        
        Args:
            error: The error that occurred
            
        Returns:
            True if we should try a fallback model
        """
        # Check for rate limit or quota errors
        if "rate_limit" in str(error).lower() or "quota" in str(error).lower():
            return True
        # Check for model availability errors
        if "model not found" in str(error).lower() or "model not available" in str(error).lower():
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "task_type": self.task_type,
            "model_provider": self.model_provider,
            "model_type": self.model_type,
            "adaptive_temperature": self.adaptive_temperature,
            "adaptive_max_tokens": self.adaptive_max_tokens
        }

class ImageModelConfig(BaseModelConfig):
    """Configuration for image generation models."""
    model: str = Field(default="stable-diffusion-xl")
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)

    @validator('model_provider')
    def set_provider(cls, v):
        return ModelProvider.STABILITY

    @validator('model_type')
    def set_type(cls, v):
        return ModelType.IMAGE

class AgentConfig(BaseModelConfig):
    """Configuration for autonomous agents."""
    base_model: str = Field(default="gpt-4")
    agent_type: str = Field(default="task")
    tools: List[str] = Field(default_factory=list)
    memory_type: str = Field(default="buffer")
    max_iterations: int = Field(default=10, ge=1, le=100)

    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ["task", "assistant", "researcher"]
        if v not in valid_types:
            raise ValueError(f"agent_type must be one of {valid_types}")
        return v

    @validator('memory_type')
    def validate_memory_type(cls, v):
        valid_types = ["buffer", "conversation", "summary"]
        if v not in valid_types:
            raise ValueError(f"memory_type must be one of {valid_types}")
        return v

    @validator('model_type')
    def set_type(cls, v):
        return ModelType.AGENT 