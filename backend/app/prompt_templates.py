from typing import Dict, Any, Optional
import json

DEFAULT_LLM_TEMPLATE = """Please provide a detailed and thoughtful response to the following prompt.
Take your time to analyze the request and provide comprehensive information.

User Input: {input}

Context Information:
{context}

Additional Instructions:
{instructions}

Please provide a well-structured, informative response that thoroughly addresses the user's input,
taking into account any relevant context provided above."""

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
            return "No additional context available."
            
        context_parts = []
        for source, data in input_data.items():
            try:
                if isinstance(data, (dict, list)):
                    formatted_data = json.dumps(data, indent=2)
                else:
                    formatted_data = str(data)
                context_parts.append(f"From {source}:\n{formatted_data}")
            except Exception as e:
                context_parts.append(f"From {source}: Error formatting data - {str(e)}")
                
        return "\n\n".join(context_parts)

    @staticmethod
    def generate_dynamic_instructions(node_config: Dict[str, Any]) -> str:
        """Generate dynamic instructions based on node configuration.
        
        Args:
            node_config: Dictionary containing node configuration
            
        Returns:
            Generated instruction string
        """
        instructions = []
        
        # Add instructions based on input/output keys
        if node_config.get("input_keys"):
            instructions.append(f"Expected inputs: {', '.join(node_config['input_keys'])}")
        if node_config.get("output_keys"):
            instructions.append(f"Expected outputs: {', '.join(node_config['output_keys'])}")
            
        # Add model-specific instructions
        if model_config := node_config.get("model_config"):
            if temp := model_config.get("temperature"):
                if temp < 0.5:
                    instructions.append("Aim for precise, factual responses")
                elif temp > 0.7:
                    instructions.append("Feel free to be more creative in your responses")
                    
        # Add format-specific instructions
        if output_format := node_config.get("output_format"):
            instructions.append(f"Format the response as: {output_format}")
            
        return "\n".join(instructions) if instructions else "Provide a clear and helpful response."

class PromptTemplate:
    def __init__(self, template: str, input_variables: Optional[list] = None):
        """Initialize a prompt template.
        
        Args:
            template: The template string with placeholders
            input_variables: List of variable names used in the template
        """
        self.template = template
        self.input_variables = input_variables or ["input", "context", "instructions"]
        self._validate_template()
        
    def _validate_template(self):
        """Validate that the template contains all required variables."""
        for var in self.input_variables:
            if "{" + var + "}" not in self.template:
                raise ValueError(f"Template missing required variable: {var}")
                
    def format(self, **kwargs) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Dictionary of variables to format the template with
            
        Returns:
            Formatted prompt string
        """
        # Ensure all required variables are provided
        missing_vars = [var for var in self.input_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
            
        return self.template.format(**kwargs)

    @classmethod
    def from_template(cls, template: str) -> 'PromptTemplate':
        """Create a PromptTemplate from a template string.
        
        Args:
            template: The template string
            
        Returns:
            New PromptTemplate instance
        """
        # Extract variables from template using basic string parsing
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return cls(template, input_variables=variables) 