from flask import Blueprint, request, jsonify, Flask, current_app
from .scriptchain import ScriptChain, LLMConfig
from .node_types import NodeFactory, LLMNode
from .prompt_templates import PromptTemplate
from typing import Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

script_chain_bp = Blueprint("script_chain", __name__)
script_chain = ScriptChain()

@script_chain_bp.route("/add_node", methods=["POST"])
def add_node():
    try:
        logger.debug("Received add_node request")
        data = request.json
        logger.debug(f"Request data: {data}")
        
        node_type = data.get("node_type", "LLM")  # Default to LLM if not specified
        logger.debug(f"Node type: {node_type}")
        
        # Validate prompt template if provided
        prompt_template = data.get("prompt_template")
        if prompt_template:
            try:
                # Validate the template format
                PromptTemplate.from_template(prompt_template)
            except ValueError as e:
                logger.error(f"Invalid prompt template: {str(e)}")
                return jsonify({
                    "error": f"Invalid prompt template: {str(e)}"
                }), 400
        
        # Create node using factory
        logger.debug("Creating node using factory")
        node = NodeFactory.create_node(
            node_type=node_type,
            node_id=data["node_id"],
            input_keys=data.get("input_keys", []),
            output_keys=data.get("output_keys", []),
            model_config=LLMConfig(**data.get("model_config", {})) if node_type == "LLM" else None,
            processing_function=data.get("processing_function"),
            decision_logic=data.get("decision_logic"),
            prompt_template=prompt_template,
            output_format=data.get("output_format")
        )
        
        logger.debug(f"Node created successfully: {node.node_id}")
        script_chain.add_node(node)
        logger.debug("Node added to script chain")
        
        response = {
            "message": f"Node {data['node_id']} added successfully",
            "node_type": node_type,
            "config": {
                "input_keys": node.input_keys,
                "output_keys": node.output_keys,
                "prompt_template": prompt_template if prompt_template else "Using default template",
                "output_format": data.get("output_format", "Not specified")
            }
        }
        logger.debug(f"Sending response: {response}")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"ValueError in add_node: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in add_node: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to add node: {str(e)}"}), 500

@script_chain_bp.route("/add_edge", methods=["POST"])
def add_edge():
    try:
        data = request.json
        script_chain.add_edge(data["from_node"], data["to_node"])
        return jsonify({
            "message": f"Edge from {data['from_node']} to {data['to_node']} added successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to add edge: {str(e)}"}), 500

@script_chain_bp.route("/execute", methods=["POST"])
def execute():
    data = request.json
    node_id = data.get("node_id")
    message = data.get("message")
    metadata = data.get("metadata", {})
    
    if not node_id or not message:
        return jsonify({"error": "Missing node_id or message"}), 400
        
    try:
        # Add execution timestamp to metadata
        metadata.update({
            "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": data.get("request_id", "")
        })
        
        response = script_chain.execute_node(node_id, message, metadata)
        node = script_chain.get_node(node_id)
        
        return jsonify({
            "response": response,
            "node_type": node.node_type,
            "context": script_chain.get_node_context(node_id, include_metadata=True)
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@script_chain_bp.route("/get_context", methods=["GET"])
def get_context():
    """Get the context for a specific node."""
    node_id = request.args.get("node_id")
    include_metadata = request.args.get("include_metadata", "false").lower() == "true"
    
    if not node_id:
        return jsonify({"error": "Missing node_id parameter"}), 400
        
    try:
        context = script_chain.get_node_context(node_id, include_metadata)
        if context is None:
            return jsonify({"error": f"No context found for node {node_id}"}), 404
            
        return jsonify({"context": context})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@script_chain_bp.route("/clear_context", methods=["POST"])
def clear_context():
    """Clear context for a specific node or all nodes."""
    data = request.json
    node_id = data.get("node_id")  # Optional
    
    try:
        script_chain.clear_context(node_id)
        message = f"Context cleared for node {node_id}" if node_id else "Context cleared for all nodes"
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@script_chain_bp.route("/update_prompt", methods=["POST"])
def update_prompt():
    """Update the prompt template for an existing node."""
    try:
        data = request.json
        node_id = data.get("node_id")
        prompt_template = data.get("prompt_template")
        
        if not node_id or not prompt_template:
            return jsonify({"error": "Missing node_id or prompt_template"}), 400
            
        node = script_chain.get_node(node_id)
        if not node:
            return jsonify({"error": f"Node {node_id} not found"}), 404
            
        if not isinstance(node, LLMNode):
            return jsonify({"error": "Only LLM nodes support custom prompt templates"}), 400
            
        # Validate and update the prompt template
        try:
            node.prompt_template = PromptTemplate.from_template(prompt_template)
            return jsonify({
                "message": f"Prompt template updated for node {node_id}",
                "template": prompt_template
            })
        except ValueError as e:
            return jsonify({"error": f"Invalid prompt template: {str(e)}"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_routes(app: Flask) -> None:
    """Register all blueprints/routes with the Flask application.
    
    Args:
        app: The Flask application instance
    """
    app.register_blueprint(script_chain_bp, url_prefix="/api/v1/scriptchain")