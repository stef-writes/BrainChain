from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dataclasses import dataclass
import os
from typing import Dict, Any, Optional
from .context_manager import ContextManager
from .model_config import LLMConfig
from .prompt_templates import PromptGenerator, DEFAULT_LLM_TEMPLATE, PromptTemplate

class ScriptChain:
    def __init__(self):
        self.nodes = {}
        self.connections = {}  # Store input/output relationships
        self.context_manager = ContextManager()

    def add_node(self, node):
        """Add a node to the chain.
        
        Args:
            node: An instance of BaseNode or its subclasses
        """
        self.nodes[node.node_id] = node
        self.connections[node.node_id] = {
            'inputs': [],
            'outputs': []
        }
        
    def get_node(self, node_id: str):
        """Get a node by its ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The node instance or None if not found
        """
        return self.nodes.get(node_id)

    def add_edge(self, from_node: str, to_node: str):
        """Add a directed edge between two nodes.
        
        Args:
            from_node: The ID of the source node
            to_node: The ID of the target node
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the chain")
            
        # Store the connection both ways
        self.connections[from_node]['outputs'].append(to_node)
        self.connections[to_node]['inputs'].append(from_node)
        
    def get_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """Get the inputs for a node from its connected input nodes.
        
        Args:
            node_id: The ID of the node to get inputs for
            
        Returns:
            dict: A dictionary mapping input node IDs to their context data
        """
        if node_id not in self.connections:
            return {}
            
        input_nodes = self.connections[node_id]['inputs']
        if not input_nodes:
            return {}
            
        return self.context_manager.get_connected_context(input_nodes)

    def execute_node(self, node_id: str, message: str, metadata: Optional[Dict] = None) -> Any:
        """Execute a specific node with a message and store its context.
        
        Args:
            node_id: The ID of the node to execute
            message: The input message for the node
            metadata: Optional metadata to store with the context
            
        Returns:
            The output from the node's execution
        """
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
            
        # Get inputs from connected nodes
        input_data = self.get_node_inputs(node_id)
        
        # Execute the node
        try:
            result = node.execute(message, input_data, self.context_manager)
            
            # Store the output in the context manager with metadata
            self.context_manager.set_context(
                node_id,
                result,
                metadata={
                    "type": node.node_type,
                    "timestamp": os.environ.get("CURRENT_TIMESTAMP", ""),
                    **(metadata or {})
                },
                input_data=message,
                connected_inputs=input_data
            )
            
            return result
        except Exception as e:
            raise Exception(f"Error executing node {node_id}: {str(e)}")

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context data for a specific node or all nodes.
        
        Args:
            node_id: The ID of the node to clear, or None to clear all
        """
        self.context_manager.clear_context(node_id)

    def get_node_context(self, node_id: str, include_metadata: bool = False) -> Any:
        """Get the stored context for a specific node.
        
        Args:
            node_id: The ID of the node
            include_metadata: Whether to include metadata in the response
            
        Returns:
            The stored context data for the node
        """
        return self.context_manager.get_context(node_id, include_metadata)

class Node:
    def __init__(self, node_id, node_type, input_keys=None, output_keys=None, model_config=None):
        self.node_id = node_id
        self.node_type = node_type
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.model_config = model_config or LLMConfig()
        self.last_output = None

    def execute(self, message: str, input_data: Dict[str, Any], context_manager: ContextManager) -> Any:
        """Execute the node with input and historical context.
        
        Args:
            message: Current input message
            input_data: Inputs from connected nodes
            context_manager: Context manager instance for accessing history
            
        Returns:
            Node output
        """
        # Get relevant historical examples
        history_records = context_manager.get_relevant_history(self.node_id, message)
        
        # Format context and history
        context = PromptGenerator.format_context(input_data)
        history = PromptGenerator.format_history(history_records)
        
        # Create prompt template
        prompt = PromptTemplate.from_template(DEFAULT_LLM_TEMPLATE)
        
        # Get API key
        api_key = os.getenv(self.model_config.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {self.model_config.api_key_env_var}")
            
        # Initialize LLM
        llm = OpenAI(
            api_key=api_key,
            model_name=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )
        
        # Create and run chain
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run(
                input=message,
                context=context,
                history=history,
                instructions=PromptGenerator.generate_dynamic_instructions({
                    "input_keys": self.input_keys,
                    "output_keys": self.output_keys,
                    "model_config": self.model_config.__dict__,
                    "output_format": getattr(self, "output_format", None)
                })
            )
            self.last_output = response.strip()
            return self.last_output
        except Exception as e:
            raise Exception(f"Error executing node {self.node_id}: {str(e)}")

    def process(self):
        # This is kept for backward compatibility
        return self.execute("Default processing", {}, ContextManager())