import json
from typing import Any, Dict, Optional

class ContextManager:
    def __init__(self):
        self.context: Dict[str, Dict[str, Any]] = {}

    def set_context(self, node_id: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Store context data for a node with optional metadata.
        
        Args:
            node_id: The ID of the node
            data: The data to store (can be any JSON-serializable type)
            metadata: Optional metadata about the context (e.g., timestamp, type)
        """
        try:
            # Ensure data is JSON-serializable
            json.dumps(data)
            
            self.context[node_id] = {
                "data": data,
                "metadata": metadata or {}
            }
        except (TypeError, ValueError) as e:
            raise ValueError(f"Context data must be JSON-serializable: {str(e)}")

    def get_context(self, node_id: str, include_metadata: bool = False) -> Any:
        """Retrieve context data for a node.
        
        Args:
            node_id: The ID of the node
            include_metadata: Whether to include metadata in the response
            
        Returns:
            The stored context data, or None if not found
        """
        if node_id not in self.context:
            return None
            
        if include_metadata:
            return self.context[node_id]
        return self.context[node_id]["data"]

    def update_context(self, node_id: str, data: Dict[str, Any]) -> None:
        """Update existing context data for a node (merge dictionaries).
        
        Args:
            node_id: The ID of the node
            data: The data to merge with existing context
        """
        if node_id in self.context and isinstance(self.context[node_id]["data"], dict):
            current_data = self.context[node_id]["data"]
            if isinstance(data, dict):
                current_data.update(data)
                self.set_context(node_id, current_data, self.context[node_id]["metadata"])
            else:
                raise ValueError("Update data must be a dictionary when updating dictionary context")
        else:
            self.set_context(node_id, data)

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context data for a specific node or all nodes.
        
        Args:
            node_id: The ID of the node to clear, or None to clear all
        """
        if node_id:
            self.context.pop(node_id, None)
        else:
            self.context.clear()

    def get_connected_context(self, node_ids: list[str]) -> Dict[str, Any]:
        """Get combined context from multiple connected nodes.
        
        Args:
            node_ids: List of node IDs to get context from
            
        Returns:
            Combined context data from all specified nodes
        """
        return {
            node_id: self.get_context(node_id)
            for node_id in node_ids
            if self.get_context(node_id) is not None
        } 