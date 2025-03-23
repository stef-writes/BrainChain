from abc import ABC, abstractmethod
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate as LangChainPrompt
from langchain.llms import OpenAI
import os
import json
from typing import Any, Dict, Optional, Type
from .model_config import BaseModelConfig, LLMConfig
from .prompt_templates import PromptTemplate, PromptGenerator, DEFAULT_LLM_TEMPLATE

class BaseNode(ABC):
    def __init__(
        self,
        node_id: str,
        node_type: str,
        model_config: Optional[BaseModelConfig] = None,
        input_keys: Optional[list] = None,
        output_keys: Optional[list] = None,
        **kwargs
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.model_config = model_config
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.last_output = None
        self.metadata = kwargs.get("metadata", {})

    @abstractmethod
    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None) -> Any:
        pass

    def _generate_context(self, input_data: Optional[Dict[str, Any]]) -> str:
        """Generate a formatted context string from input data.
        
        Args:
            input_data: Dictionary of input data from connected nodes
            
        Returns:
            A formatted context string
        """
        if not input_data:
            return ""
            
        context_parts = []
        for node_id, data in input_data.items():
            try:
                if isinstance(data, (dict, list)):
                    formatted_data = json.dumps(data, indent=2)
                else:
                    formatted_data = str(data)
                context_parts.append(f"From {node_id}:\n{formatted_data}")
            except Exception as e:
                context_parts.append(f"From {node_id}: Error formatting data - {str(e)}")
                
        return "\n\n".join(context_parts) if context_parts else ""

    def validate_api_key(self) -> bool:
        """Validate that the required API key is available."""
        if not self.model_config:
            return True  # No API key needed
            
        api_key = os.getenv(self.model_config.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.model_config.api_key_env_var}")
        return True

    @property
    def config_summary(self) -> Dict[str, Any]:
        """Get a summary of the node's configuration."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "model_type": self.model_config.model_type if self.model_config else None,
            "model_provider": self.model_config.model_provider if self.model_config else None,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "metadata": self.metadata
        }

class LLMNode(BaseNode):
    def __init__(
        self,
        node_id: str,
        model_config: Optional[LLMConfig] = None,
        prompt_template: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            node_id=node_id,
            node_type="LLM",
            model_config=model_config or LLMConfig(),
            **kwargs
        )
        self.output_format = output_format
        
        # Initialize prompt template
        try:
            self.prompt_template = PromptTemplate.from_template(prompt_template) if prompt_template else PromptTemplate(DEFAULT_LLM_TEMPLATE)
        except ValueError as e:
            # If custom template is invalid, fall back to default
            print(f"Warning: Invalid prompt template for node {node_id}: {str(e)}. Using default template.")
            self.prompt_template = PromptTemplate(DEFAULT_LLM_TEMPLATE)

    def _generate_prompt(self, message: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt using the template and available data.
        
        Args:
            message: The input message
            input_data: Optional dictionary of input data from connected nodes
            
        Returns:
            Formatted prompt string
        """
        # Generate dynamic instructions based on node configuration
        node_config = {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "model_config": self.model_config.__dict__,
            "output_format": self.output_format
        }
        instructions = PromptGenerator.generate_dynamic_instructions(node_config)
        
        # Format the prompt template
        try:
            return self.prompt_template.format(
                input=message,
                context=self._generate_context(input_data),
                instructions=instructions
            )
        except ValueError as e:
            # If formatting fails, use a simplified prompt
            return f"Input: {message}\nContext: {self._generate_context(input_data)}\nInstructions: {instructions}"

    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        self.validate_api_key()
        # Generate the prompt
        prompt_text = self._generate_prompt(message, input_data)
        
        # Create LangChain prompt template
        prompt = LangChainPrompt(
            input_variables=["prompt"],
            template="{prompt}"
        )

        llm = OpenAI(
            api_key=os.getenv(self.model_config.api_key_env_var),
            model_name=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run(prompt=prompt_text)
            self.last_output = response.strip()
            return self.last_output
        except Exception as e:
            raise Exception(f"Error executing node {self.node_id}: {str(e)}")

class DataProcessingNode(BaseNode):
    def __init__(self, node_id: str, processing_function: Optional[callable] = None, **kwargs):
        super().__init__(node_id, node_type="DataProcessing", **kwargs)
        self.processing_function = processing_function

    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None) -> Any:
        try:
            if self.processing_function:
                result = self.processing_function(message, input_data)
            else:
                # Default behavior: structure the input as JSON if possible
                try:
                    result = json.loads(message)
                except json.JSONDecodeError:
                    result = {"message": message, "input_data": input_data}
                    
            self.last_output = result
            return result
        except Exception as e:
            raise Exception(f"Error in data processing node {self.node_id}: {str(e)}")

class DecisionNode(BaseNode):
    def __init__(self, node_id: str, decision_logic: Optional[callable] = None, **kwargs):
        super().__init__(node_id, node_type="Decision", **kwargs)
        self.decision_logic = decision_logic

    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if self.decision_logic:
                result = self.decision_logic(message, input_data)
            else:
                # Default behavior: return a structured decision result
                result = {
                    "decision": "default",
                    "message": message,
                    "input_data": input_data,
                    "reason": "No decision logic provided"
                }
                
            self.last_output = result
            return result
        except Exception as e:
            raise Exception(f"Error in decision node {self.node_id}: {str(e)}")

class NodeFactory:
    _node_types = {
        "LLM": LLMNode,
        "DataProcessing": DataProcessingNode,
        "Decision": DecisionNode
    }

    @classmethod
    def register_node_type(cls, type_name: str, node_class: Type[BaseNode]):
        """Register a new node type.
        
        Args:
            type_name: The name of the node type
            node_class: The node class to register
        """
        if not issubclass(node_class, BaseNode):
            raise ValueError(f"Node class must inherit from BaseNode")
        cls._node_types[type_name] = node_class

    @classmethod
    def create_node(cls, node_type: str, node_id: str, **kwargs) -> BaseNode:
        """Create a node instance based on the specified type.
        
        Args:
            node_type: The type of node to create
            node_id: The ID for the new node
            **kwargs: Additional arguments for node initialization
            
        Returns:
            An instance of the specified node type
            
        Raises:
            ValueError: If the node type is unknown
        """
        node_class = cls._node_types.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")
            
        return node_class(node_id, **kwargs) 