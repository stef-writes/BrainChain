import React, { useState, useCallback } from 'react';
import ReactFlow, {
  addEdge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import ChatGPTNode from './components/Node';
import 'reactflow/dist/style.css';

// Define node types
const nodeTypes = {
  chatGPT: ChatGPTNode,
};

console.log('Registered node types:', nodeTypes);

const initialNodes = [];
const initialEdges = [];

const ScriptChainFlow = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [nodeResponses, setNodeResponses] = useState({});
  const [nodeNames, setNodeNames] = useState({});

  // Debug current state
  React.useEffect(() => {
    console.log('Current nodes:', nodes);
    console.log('Current edges:', edges);
  }, [nodes, edges]);

  const getNodeSources = useCallback((nodeId) => {
    return edges
      .filter(edge => edge.target === nodeId)
      .map(edge => edge.source);
  }, [edges]);

  const updateNodeData = useCallback(() => {
    console.log('Updating node data...');
    setNodes((nds) =>
      nds.map((node) => {
        console.log('Processing node:', node);
        return {
          ...node,
          data: {
            ...node.data,
            sourceNodes: getNodeSources(node.id),
            responses: nodeResponses,
            nodeName: nodeNames[node.id],
            onResponse: (nodeId, response) => {
              setNodeResponses(prev => ({
                ...prev,
                [nodeId]: response
              }));
            },
            onNameChange: (nodeId, name) => {
              setNodeNames(prev => ({
                ...prev,
                [nodeId]: name
              }));
            }
          },
        };
      })
    );
  }, [getNodeSources, nodeResponses, nodeNames]);

  // Update node data when edges or responses change
  React.useEffect(() => {
    updateNodeData();
  }, [edges, nodeResponses, nodeNames, updateNodeData]);

  const onConnect = useCallback((params) => {
    console.log('Connecting nodes:', params);
    setEdges((eds) => addEdge(params, eds));
    fetch(`${process.env.REACT_APP_API_URL}/add_edge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from_node: params.source,
        to_node: params.target,
      }),
    });
  }, [setEdges]);

  const addNode = async () => {
    console.log('Add node button clicked!');
    const nodeId = `node-${nodes.length + 1}`;
    console.log('Creating new node with ID:', nodeId);
    console.log('Current nodes:', nodes);
    
    // Try to create node in backend first
    try {
      console.log('Sending request to backend...');
      const response = await fetch(`${process.env.REACT_APP_API_URL}/add_node`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          node_id: nodeId,
          node_type: 'LLM',
          model_config: {
            model: "gpt-4",
            temperature: 0.7,
            max_tokens: 2000,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            model_provider: "openai",
            model_type: "text",
            api_key_env_var: "OPENAI_API_KEY"
          }
        }),
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create node in backend');
      }

      // Only create node in UI if backend creation succeeded
      const newNode = {
        id: nodeId,
        type: 'chatGPT',  // This must match the key in nodeTypes
        position: {
          x: Math.random() * 500,
          y: Math.random() * 500,
        },
        data: {
          label: `Node ${nodes.length + 1}`,
          sourceNodes: [],
          responses: nodeResponses,
          onResponse: (nodeId, response) => {
            setNodeResponses(prev => ({
              ...prev,
              [nodeId]: response
            }));
          },
          onNameChange: (nodeId, name) => {
            setNodeNames(prev => ({
              ...prev,
              [nodeId]: name
            }));
          }
        },
      };

      console.log('New node configuration:', newNode);
      console.log('Adding node to frontend state...');
      setNodes((nds) => {
        const updatedNodes = nds.concat(newNode);
        console.log('Updated nodes:', updatedNodes);
        return updatedNodes;
      });
      
    } catch (error) {
      console.error('Error creating node:', error);
      alert(`Failed to create node: ${error.message}`);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div className="workspace-controls">
        <div className="control-group">
          <button onClick={addNode} className="add-node-button">
            <span className="button-icon">+</span>
            Create New Node
          </button>
        </div>
        <div className="control-group">
          <button 
            onClick={() => setNodes([])} 
            className="clear-all-button"
            disabled={nodes.length === 0}
          >
            Clear All Nodes
          </button>
        </div>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        className="react-flow-wrapper"
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default ScriptChainFlow;