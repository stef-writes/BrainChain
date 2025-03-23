from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_LLM_TEMPLATE = """Please provide a response to the following input, taking into account any relevant context and historical examples.

Input: {input}

Context Information:
{context}

Relevant Historical Examples:
{history}

Additional Instructions:
{instructions}

Please provide a clear and helpful response that addresses the input directly."""

class PromptGenerator:
    @staticmethod
    def format_context(input_data: Optional[Dict[str, Any]] = None) -> str:
        """Format input data into a readable context string.
        
        Args:
            input_data: Dictionary of input data from connected nodes
            
        Returns:
            Formatted context string
        """
        if not input_data:
            logger.debug("No input data provided for context formatting")
            return "No additional context available."
            
        logger.info("Formatting context from input data")
        logger.debug(f"Input data: {input_data}")
            
        context_parts = []
        for source, data in input_data.items():
            try:
                if isinstance(data, (dict, list)):
                    formatted_data = json.dumps(data, indent=2)
                else:
                    formatted_data = str(data)
                context_parts.append(f"From {source}:\n{formatted_data}")
                logger.debug(f"Formatted data from {source}")
            except Exception as e:
                logger.error(f"Error formatting data from {source}: {str(e)}")
                context_parts.append(f"From {source}: Error formatting data - {str(e)}")
                
        formatted_context = "\n\n".join(context_parts)
        logger.debug(f"Final formatted context: {formatted_context}")
        return formatted_context

    @staticmethod
    def format_history(history_records: List[Any]) -> str:
        """Format historical execution records into a readable string.
        
        Args:
            history_records: List of historical execution records
            
        Returns:
            Formatted history string
        """
        if not history_records:
            logger.debug("No history records provided for formatting")
            return "No relevant historical examples available."
            
        logger.info(f"Formatting {len(history_records)} history records")
        history_parts = []
        
        for i, record in enumerate(history_records):
            try:
                logger.debug(f"Processing history record {i+1}/{len(history_records)}")
                logger.debug(f"Record data: {record}")
                
                # Format connected inputs if available
                connected_inputs = ""
                if hasattr(record, 'connected_inputs') and record.connected_inputs:
                    logger.debug(f"Formatting connected inputs for record: {record.connected_inputs}")
                    connected_inputs = "\nConnected Inputs:\n" + "\n".join(
                        f"- {node_id}: {data}"
                        for node_id, data in record.connected_inputs.items()
                    )
                    logger.debug(f"Formatted connected inputs: {connected_inputs}")
                
                # Ensure record has required attributes
                if not hasattr(record, 'input') or not hasattr(record, 'output'):
                    logger.error(f"Record {i+1} missing required attributes: input={hasattr(record, 'input')}, output={hasattr(record, 'output')}")
                    continue
                
                formatted_record = (
                    f"Previous Example:\n"
                    f"Input: {record.input}\n"
                    f"Output: {record.output}{connected_inputs}"
                )
                history_parts.append(formatted_record)
                logger.debug(f"Formatted history record {i+1}: {formatted_record}")
            except Exception as e:
                logger.error(f"Error formatting historical record {i+1}: {str(e)}")
                history_parts.append(f"Error formatting historical record: {str(e)}")
                
        formatted_history = "\n\n".join(history_parts)
        logger.debug(f"Final formatted history: {formatted_history}")
        return formatted_history

    @staticmethod
    def generate_dynamic_instructions(node_config: Dict[str, Any]) -> str:
        """Generate dynamic instructions based on node configuration.
        
        Args:
            node_config: Dictionary containing node configuration
            
        Returns:
            Generated instruction string
        """
        logger.info("Generating dynamic instructions from node config")
        logger.debug(f"Node config: {node_config}")
        
        instructions = []
        
        # Add instructions based on input/output keys
        if node_config.get("input_keys"):
            instructions.append(f"Expected inputs: {', '.join(node_config['input_keys'])}")
            logger.debug(f"Added input key instructions: {instructions[-1]}")
        if node_config.get("output_keys"):
            instructions.append(f"Expected outputs: {', '.join(node_config['output_keys'])}")
            logger.debug(f"Added output key instructions: {instructions[-1]}")
            
        # Add model-specific instructions
        if model_config := node_config.get("model_config"):
            if temp := model_config.get("temperature"):
                if temp < 0.5:
                    instructions.append("Aim for precise, factual responses")
                elif temp > 0.7:
                    instructions.append("Feel free to be more creative in your responses")
                logger.debug(f"Added temperature-based instructions: {instructions[-1]}")
                    
        # Add format-specific instructions
        if output_format := node_config.get("output_format"):
            instructions.append(f"Format the response as: {output_format}")
            logger.debug(f"Added format instructions: {instructions[-1]}")
            
        final_instructions = "\n".join(instructions) if instructions else "Provide a clear and helpful response."
        logger.debug(f"Final generated instructions: {final_instructions}")
        return final_instructions

class PromptTemplate:
    def __init__(self, template: str, input_variables: Optional[list] = None):
        """Initialize a prompt template.
        
        Args:
            template: The template string with placeholders
            input_variables: List of variable names used in the template
        """
        self.template = template
        self.input_variables = input_variables or ["input", "context", "history", "instructions"]
        logger.info(f"Initializing PromptTemplate with variables: {self.input_variables}")
        self._validate_template()
        
    def _validate_template(self):
        """Validate that the template contains all required variables."""
        logger.debug("Validating template variables")
        for var in self.input_variables:
            if "{" + var + "}" not in self.template:
                logger.error(f"Template missing required variable: {var}")
                raise ValueError(f"Template missing required variable: {var}")
        logger.debug("Template validation successful")
                
    def format(self, **kwargs) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Dictionary of variables to format the template with
            
        Returns:
            Formatted prompt string
        """
        logger.info("Formatting prompt template")
        logger.debug(f"Template variables: {kwargs}")
        
        # Ensure all required variables are provided
        missing_vars = [var for var in self.input_variables if var not in kwargs]
        if missing_vars:
            logger.error(f"Missing required variables: {missing_vars}")
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
            
        formatted_prompt = self.template.format(**kwargs)
        logger.debug(f"Formatted prompt: {formatted_prompt}")
        return formatted_prompt

    @classmethod
    def from_template(cls, template: str) -> 'PromptTemplate':
        """Create a PromptTemplate from a template string.
        
        Args:
            template: The template string
            
        Returns:
            New PromptTemplate instance
        """
        logger.info("Creating PromptTemplate from template string")
        # Extract variables from template using basic string parsing
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        logger.debug(f"Extracted variables from template: {variables}")
        return cls(template, input_variables=variables) 