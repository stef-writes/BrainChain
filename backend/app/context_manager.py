import json
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
from .models import ExecutionHistory
from . import db
import uuid
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import TokenUsage
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Try importing sentence-transformers, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Semantic search will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class ExecutionMetadata(BaseModel):
    """Metadata for execution records."""
    type: str
    timestamp: str
    additional_info: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class ExecutionRecord:
    """Record of a node execution."""
    node_id: str
    input: str
    output: str
    timestamp: datetime
    metadata: Dict[str, Any]
    connected_inputs: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None
    compressed: bool = False

class ContextManager:
    def __init__(self, max_history: int = 10, batch_size: int = 5, cache_size: int = 1000):
        """Initialize the context manager.
        
        Args:
            max_history: Maximum number of historical records to keep
            batch_size: Size of batches for processing records
            cache_size: Maximum size of the embedding cache
        """
        self.max_history = max_history
        self.batch_size = batch_size
        self.history: List[ExecutionRecord] = []
        self.token_usage = TokenUsage()
        self.model = None
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache = {}
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize sentence transformer for semantic search if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer model")
            except Exception as e:
                logger.error(f"Failed to initialize sentence transformer: {str(e)}")
                self.model = None
        else:
            logger.info("Semantic search disabled - sentence-transformers not available")

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache or generate new one.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding array or None if generation fails
        """
        if not self.model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def _process_batch(self, records: List[ExecutionRecord]) -> List[ExecutionRecord]:
        """Process a batch of records in parallel.
        
        Args:
            records: List of records to process
            
        Returns:
            Processed records with embeddings
        """
        if not self.model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return records
            
        futures = []
        for record in records:
            if not record.embedding:
                futures.append(
                    self._executor.submit(self._get_cached_embedding, record.input)
                )
                
        for record, future in zip(records, futures):
            try:
                record.embedding = future.result()
            except Exception as e:
                logger.error(f"Error processing record: {str(e)}")
                
        return records

    def _compress_history(self, records: List[ExecutionRecord]) -> List[ExecutionRecord]:
        """Compress history records to fit within token limits.
        
        Args:
            records: List of records to compress
            
        Returns:
            Compressed list of records
        """
        if not records:
            return []
            
        # Sort by timestamp
        records.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Calculate total tokens
        total_tokens = sum(self.token_usage.estimate_tokens(str(r)) for r in records)
        
        # If within limits, return as is
        if total_tokens <= self.token_usage.max_tokens:
            return records
            
        # Compress records
        compressed_records = []
        current_tokens = 0
        
        for record in records:
            record_tokens = self.token_usage.estimate_tokens(str(record))
            
            # If adding this record would exceed limit, compress it
            if current_tokens + record_tokens > self.token_usage.max_tokens:
                if not record.compressed:
                    # Compress the record
                    record.input = self._compress_text(record.input)
                    record.output = self._compress_text(record.output)
                    record.compressed = True
                    record_tokens = self.token_usage.estimate_tokens(str(record))
                    
            if current_tokens + record_tokens <= self.token_usage.max_tokens:
                compressed_records.append(record)
                current_tokens += record_tokens
            else:
                break
                
        return compressed_records

    def _compress_text(self, text: str) -> str:
        """Compress text while preserving key information.
        
        Args:
            text: Text to compress
            
        Returns:
            Compressed text
        """
        # Simple compression - keep first and last sentences
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text
            
        return f"{sentences[0]}. ... {sentences[-1]}"

    def add_record(self, record: ExecutionRecord) -> None:
        """Add an execution record to history.
        
        Args:
            record: The execution record to add
        """
        with self._lock:
            # Process in batches if needed
            if len(self.history) >= self.batch_size:
                self.history = self._process_batch(self.history)
                
            # Add new record
            self.history.append(record)
            
            # Maintain max history size
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            # Optimize context if needed
            if self.token_usage.estimate_tokens(str(self.history)) > self.token_usage.max_tokens:
                self.history = self._compress_history(self.history)

    def get_relevant_history(self, node_id: str, current_input: str, 
                           max_examples: int = 3) -> List[ExecutionRecord]:
        """Get relevant historical executions for in-context learning.
        
        Args:
            node_id: ID of the node
            current_input: Current input to find relevant history for
            max_examples: Maximum number of historical examples to return
            
        Returns:
            List of relevant execution records
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")
            
        logger.info(f"Getting relevant history for node {node_id}")
        logger.debug(f"Current input: {current_input}")
        logger.debug(f"Max examples: {max_examples}")
        
        # Filter records for the specific node
        node_records = [r for r in self.history if r.node_id == node_id]
        if not node_records:
            return []
            
        # Process records in batches if needed
        if len(node_records) > self.batch_size:
            node_records = self._process_batch(node_records)
            
        # If we have embeddings and sentence-transformers is available, use semantic search
        if self.model and SENTENCE_TRANSFORMERS_AVAILABLE and all(r.embedding is not None for r in node_records):
            try:
                current_embedding = self._get_cached_embedding(current_input)
                if current_embedding is not None:
                    similarities = [
                        cosine_similarity([current_embedding], [r.embedding])[0][0]
                        for r in node_records
                    ]
                    # Sort by similarity and get top examples
                    relevant_records = [
                        record for _, record in sorted(
                            zip(similarities, node_records),
                            reverse=True
                        )
                    ][:max_examples]
                    return relevant_records
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                
        # Fallback to recency-based selection
        return sorted(
            node_records,
            key=lambda x: x.timestamp,
            reverse=True
        )[:max_examples]

    def summarize_context(self, node_id: str, max_tokens: Optional[int] = None) -> str:
        """Generate a summary of the context for a node.
        
        Args:
            node_id: ID of the node to summarize context for
            max_tokens: Optional maximum tokens for the summary
            
        Returns:
            Summarized context string
        """
        records = [r for r in self.history if r.node_id == node_id]
        if not records:
            return "No historical context available."
            
        # Sort by timestamp
        records.sort(key=lambda x: x.timestamp)
        
        # Create summary
        summary_parts = []
        for record in records:
            summary_parts.append(
                f"Input: {record.input}\n"
                f"Output: {record.output}\n"
                f"Time: {record.timestamp}\n"
            )
            
        summary = "\n".join(summary_parts)
        
        # Optimize if needed
        if max_tokens:
            summary = self.token_usage.optimize_context(summary, max_tokens)
            
        return summary

    def clear_history(self, node_id: Optional[str] = None) -> None:
        """Clear history for a specific node or all nodes.
        
        Args:
            node_id: Optional ID of the node to clear history for
        """
        if node_id:
            self.history = [r for r in self.history if r.node_id != node_id]
        else:
            self.history = []

    def set_context(self, node_id: str, output: Any, metadata: Optional[Dict[str, Any]] = None, 
                   input_data: Optional[str] = None, connected_inputs: Optional[Dict[str, Any]] = None) -> None:
        """Store context and execution record for a node.
        
        Args:
            node_id: ID of the node
            output: Output from the node execution
            metadata: Additional metadata about the execution
            input_data: Input that triggered the execution
            connected_inputs: Inputs from connected nodes
            
        Raises:
            ValueError: If node_id is empty or invalid
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        logger.info(f"Setting context for node {node_id}")
        logger.debug(f"Input data: {input_data}")
        logger.debug(f"Output: {output}")
        logger.debug(f"Metadata: {metadata}")
        logger.debug(f"Connected inputs: {connected_inputs}")

        # Validate and sanitize data
        try:
            sanitized_output = self._sanitize_data(output)
            sanitized_input = self._sanitize_data(input_data) if input_data else ""
            sanitized_metadata = self._sanitize_data(metadata) if metadata else {}
            sanitized_connected_inputs = self._sanitize_data(connected_inputs) if connected_inputs else {}
        except Exception as e:
            logger.error(f"Error sanitizing data: {str(e)}")
            raise ValueError(f"Invalid data format: {str(e)}")

        # Store current context in memory
        self.contexts[node_id] = {
            "output": sanitized_output,
            "metadata": ExecutionMetadata(
                type=sanitized_metadata.get("type", "unknown"),
                timestamp=datetime.now().isoformat(),
                additional_info=sanitized_metadata
            ).dict(),
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Stored context in memory: {self.contexts[node_id]}")

        # Store execution record in database
        try:
            history_record = ExecutionHistory(
                id=str(uuid.uuid4()),
                node_id=node_id,
                input_data=sanitized_input,
                output_data=str(sanitized_output),
                execution_metadata=sanitized_metadata,
                connected_inputs=sanitized_connected_inputs
            )
            db.session.add(history_record)
            db.session.commit()
            logger.info(f"Stored execution record in database with ID: {history_record.id}")
            
            # Maintain max history size in database
            history_count = ExecutionHistory.query.filter_by(node_id=node_id).count()
            if history_count > self.max_history:
                oldest_record = ExecutionHistory.query.filter_by(node_id=node_id).order_by(ExecutionHistory.created_at).first()
                if oldest_record:
                    db.session.delete(oldest_record)
                    db.session.commit()
                    logger.info(f"Removed oldest record {oldest_record.id} to maintain max history size")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error storing execution history: {str(e)}")
            raise Exception(f"Error storing execution history: {str(e)}")

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data to ensure it can be safely stored.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
            
        Raises:
            ValueError: If data cannot be sanitized
        """
        if data is None:
            return None
            
        if isinstance(data, (str, int, float, bool)):
            return data
            
        if isinstance(data, (list, tuple)):
            return [self._sanitize_data(item) for item in data]
            
        if isinstance(data, dict):
            return {str(k): self._sanitize_data(v) for k, v in data.items()}
            
        try:
            return str(data)
        except Exception as e:
            raise ValueError(f"Cannot sanitize data: {str(e)}")

    def get_context(self, node_id: str, include_metadata: bool = False) -> Union[Any, Dict[str, Any]]:
        """Get the current context for a node.
        
        Args:
            node_id: ID of the node
            include_metadata: Whether to include metadata in response
            
        Returns:
            Current context data or None if not found
            
        Raises:
            ValueError: If node_id is empty or invalid
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")
            
        logger.info(f"Getting context for node {node_id}")
        context = self.contexts.get(node_id)
        if not context:
            logger.warning(f"No context found for node {node_id}")
            return None
            
        logger.debug(f"Retrieved context: {context}")
        if include_metadata:
            return context
        return context.get("output")

    def get_connected_context(self, node_ids: List[str]) -> Dict[str, Any]:
        """Get context from connected nodes.
        
        Args:
            node_ids: List of connected node IDs
            
        Returns:
            Dictionary mapping node IDs to their context
        """
        if not node_ids:
            logger.warning("No node IDs provided for connected context")
            return {}
            
        logger.info(f"Getting connected context for nodes: {node_ids}")
        connected_context = {
            node_id: self.get_context(node_id)
            for node_id in node_ids
            if node_id in self.contexts
        }
        logger.debug(f"Retrieved connected context: {connected_context}")
        return connected_context

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context and history for a node or all nodes.
        
        Args:
            node_id: ID of node to clear, or None to clear all
            
        Raises:
            ValueError: If node_id is provided but invalid
        """
        if node_id is not None and not isinstance(node_id, str):
            raise ValueError("node_id must be a string if provided")
            
        try:
            with self._lock:
                if node_id:
                    logger.info(f"Clearing context for node {node_id}")
                    # Clear memory context
                    self.contexts.pop(node_id, None)
                    # Clear database history
                    ExecutionHistory.query.filter_by(node_id=node_id).delete()
                    # Clear history records
                    self.history = [r for r in self.history if r.node_id != node_id]
                else:
                    logger.info("Clearing all context")
                    # Clear all memory context
                    self.contexts.clear()
                    # Clear all database history
                    ExecutionHistory.query.delete()
                    # Clear all history records
                    self.history.clear()
                db.session.commit()
                logger.info("Successfully cleared context")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error clearing context: {str(e)}")
            raise Exception(f"Error clearing context: {str(e)}")

    def __del__(self):
        """Cleanup when the context manager is destroyed."""
        self._executor.shutdown(wait=True) 