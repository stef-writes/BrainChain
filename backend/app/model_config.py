from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class BaseModelConfig:
    """Base configuration class for all model types."""
    model_provider: str  # e.g., "openai", "anthropic", "stability-ai"
    model_type: str     # e.g., "text", "image", "agent"
    api_key_env_var: str = "OPENAI_API_KEY"  # Default to OpenAI
    additional_params: Dict[str, Any] = None

@dataclass
class LLMConfig(BaseModelConfig):
    """Configuration for language models."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        self.model_provider = "openai"  # Default provider
        self.model_type = "text"
        if self.model.startswith("claude"):
            self.model_provider = "anthropic"
            self.api_key_env_var = "ANTHROPIC_API_KEY"

@dataclass
class ImageModelConfig(BaseModelConfig):
    """Configuration for image generation models."""
    model: str = "stable-diffusion-xl"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        self.model_provider = "stability-ai"
        self.model_type = "image"
        self.api_key_env_var = "STABILITY_API_KEY"

@dataclass
class AgentConfig(BaseModelConfig):
    """Configuration for autonomous agents."""
    base_model: str = "gpt-4"
    agent_type: str = "task"  # e.g., "task", "assistant", "researcher"
    tools: list = None
    memory_type: str = "buffer"
    max_iterations: int = 10

    def __post_init__(self):
        self.model_provider = "openai"  # Default provider
        self.model_type = "agent"
        self.tools = self.tools or [] 