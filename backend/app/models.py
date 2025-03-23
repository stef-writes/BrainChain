from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Table, Text
from sqlalchemy.orm import relationship, validates
from datetime import datetime
from typing import List, Dict, Any
import json
from . import db

# Association table for node connections
node_connections = Table(
    'node_connections',
    db.Model.metadata,
    Column('from_node_id', String(100), ForeignKey('nodes.node_id'), primary_key=True),
    Column('to_node_id', String(100), ForeignKey('nodes.node_id'), primary_key=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)

class Node(db.Model):
    __tablename__ = 'nodes'
    
    node_id = Column(String(100), primary_key=True)
    node_type = Column(String(100), nullable=False)
    input_keys = Column(JSON, default=list)
    output_keys = Column(JSON, default=list)
    model_config = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    input_connections = relationship(
        'Node',
        secondary=node_connections,
        primaryjoin=(node_connections.c.to_node_id == node_id),
        secondaryjoin=(node_connections.c.from_node_id == node_id),
        backref='output_connections'
    )
    execution_history = relationship('ExecutionHistory', back_populates='node', cascade='all, delete-orphan')

    @validates('node_id')
    def validate_node_id(self, key, value):
        if not value or not isinstance(value, str):
            raise ValueError("node_id must be a non-empty string")
        return value

    @validates('node_type')
    def validate_node_type(self, key, value):
        valid_types = ['LLM', 'DataProcessing', 'Decision', 'ImageGeneration']
        if value not in valid_types:
            raise ValueError(f"node_type must be one of {valid_types}")
        return value

    @validates('input_keys', 'output_keys')
    def validate_keys(self, key, value):
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list")
        return value

    @validates('model_config')
    def validate_model_config(self, key, value):
        if not isinstance(value, dict):
            raise ValueError("model_config must be a dictionary")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation.
        
        Returns:
            Dictionary containing node data
        """
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'input_keys': self.input_keys,
            'output_keys': self.output_keys,
            'model_config': self.model_config,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'input_connections': [node.node_id for node in self.input_connections],
            'output_connections': [node.node_id for node in self.output_connections]
        }

    def __repr__(self) -> str:
        """String representation of the node.
        
        Returns:
            String representation
        """
        return f"<Node(node_id='{self.node_id}', node_type='{self.node_type}')>"

class ExecutionHistory(db.Model):
    __tablename__ = 'execution_history'
    
    id = Column(String(100), primary_key=True)
    node_id = Column(String(100), ForeignKey('nodes.node_id'), nullable=False)
    input_data = Column(Text)
    output_data = Column(Text)
    execution_metadata = Column(JSON)
    connected_inputs = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    node = relationship('Node', back_populates='execution_history')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution history to dictionary representation.
        
        Returns:
            Dictionary containing execution history data
        """
        return {
            'id': self.id,
            'node_id': self.node_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'metadata': self.execution_metadata,
            'connected_inputs': self.connected_inputs,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }