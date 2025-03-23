from abc import ABC, abstractmethod
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate as LangChainPrompt
from langchain.llms import OpenAI
import os
import json
from typing import Any, Dict, Optional, Type
from .model_config import BaseModelConfig, LLMConfig
from .prompt_templates import PromptTemplate, PromptGenerator, DEFAULT_LLM_TEMPLATE
from .context_manager import ContextManager
import logging

logger = logging.getLogger(__name__)

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
    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None, context_manager: Optional[ContextManager] = None) -> Any:
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
            self.prompt_template = PromptTemplate.from_template(prompt_template) if prompt_template else PromptTemplate.from_template(DEFAULT_LLM_TEMPLATE)
        except ValueError as e:
            # If custom template is invalid, fall back to default
            print(f"Warning: Invalid prompt template for node {node_id}: {str(e)}. Using default template.")
            self.prompt_template = PromptTemplate.from_template(DEFAULT_LLM_TEMPLATE)

    def execute(self, message: str, input_data: Optional[Dict[str, Any]] = None, context_manager: Optional[ContextManager] = None) -> str:
        """Execute the LLM node with input and historical context.
        
        Args:
            message: Current input message
            input_data: Inputs from connected nodes
            context_manager: Context manager instance for accessing history
            
        Returns:
            Node output
        """
        logger.info(f"Executing LLM node {self.node_id}")
        logger.debug(f"Input message: {message}")
        logger.debug(f"Input data: {input_data}")
        
        self.validate_api_key()
        
        # Get relevant historical examples if context manager is provided
        history_records = []
        if context_manager:
            logger.debug("Context manager provided, getting relevant history")
            try:
                history_records = context_manager.get_relevant_history(self.node_id, message)
                logger.debug(f"Retrieved {len(history_records)} history records")
            except Exception as e:
                logger.error(f"Error getting history records: {str(e)}")
                history_records = []
        
        # Format context and history
        logger.debug("Formatting context and history")
        context = self._generate_context(input_data)
        logger.debug(f"Formatted context: {context}")
        
        try:
            history = PromptGenerator.format_history(history_records)
            logger.debug(f"Formatted history: {history}")
        except Exception as e:
            logger.error(f"Error formatting history: {str(e)}")
            history = "No relevant historical examples available."
        
        # Generate dynamic instructions
        logger.debug("Generating dynamic instructions")
        node_config = {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "model_config": self.model_config.__dict__,
            "output_format": self.output_format
        }
        logger.debug(f"Node config: {node_config}")
        
        try:
            instructions = PromptGenerator.generate_dynamic_instructions(node_config)
            logger.debug(f"Generated instructions: {instructions}")
        except Exception as e:
            logger.error(f"Error generating instructions: {str(e)}")
            instructions = "Provide a clear and helpful response."
        
        # Format the prompt
        logger.debug("Formatting prompt template")
        try:
            prompt_text = self.prompt_template.format(
                input=message,
                context=context,
                history=history,
                instructions=instructions
            )
            logger.debug(f"Formatted prompt: {prompt_text}")
        except ValueError as e:
            logger.error(f"Error formatting prompt template: {str(e)}")
            # If formatting fails, use a simplified prompt
            prompt_text = f"Input: {message}\nContext: {context}\nHistory: {history}\nInstructions: {instructions}"
            logger.debug(f"Using simplified prompt: {prompt_text}")

        # Create LangChain prompt template
        logger.debug("Creating LangChain prompt template")
        langchain_prompt = LangChainPrompt(
            input_variables=["text"],
            template="{text}"
        )

        logger.debug("Initializing OpenAI LLM")
        llm = OpenAI(
            api_key=os.getenv(self.model_config.api_key_env_var),
            model_name=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )

        logger.debug("Creating and running LLMChain")
        chain = LLMChain(llm=llm, prompt=langchain_prompt)
        try:
            response = chain.run(text=prompt_text)
            self.last_output = response.strip()
            logger.info(f"Successfully executed node {self.node_id}")
            logger.debug(f"Response: {self.last_output}")
            return self.last_output
        except Exception as e:
            logger.error(f"Error in LLMChain execution: {str(e)}")
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