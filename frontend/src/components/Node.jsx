import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Handle, Position } from 'reactflow';

const debounce = (fn, ms) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

const ChatGPTNode = ({ data, id }) => {
  const [nodeName, setNodeName] = useState(data.label || 'New Node');
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showVariableMenu, setShowVariableMenu] = useState(false);
  const [promptHistory, setPromptHistory] = useState([]);
  const textAreaRef = useRef(null);
  const responseAreaRef = useRef(null);
  const variableMenuRef = useRef(null);

  useEffect(() => {
    // Update node name in parent when it changes
    if (data.onNameChange) {
      data.onNameChange(id, nodeName);
    }
  }, [nodeName, id, data]);

  // Close variable menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (variableMenuRef.current && !variableMenuRef.current.contains(event.target)) {
        setShowVariableMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle textarea resizing
  useEffect(() => {
    const textArea = textAreaRef.current;
    const responseArea = responseAreaRef.current;
    
    if (!textArea || !responseArea) return;

    const resizeObserver = new ResizeObserver(
      debounce(() => {
        // Force a reflow to ensure proper layout
        textArea.style.height = 'auto';
        responseArea.style.height = 'auto';
      }, 100)
    );

    resizeObserver.observe(textArea);
    resizeObserver.observe(responseArea);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  const insertVariable = (sourceNodeId) => {
    const textarea = textAreaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const currentValue = textarea.value;
    
    // Insert the variable placeholder
    const newValue = 
      currentValue.substring(0, start) +
      `{${sourceNodeId}}` +
      currentValue.substring(end);
    
    setMessage(newValue);
    setShowVariableMenu(false);

    // Reset cursor position after state update
    setTimeout(() => {
      textarea.focus();
      const newPosition = start + sourceNodeId.length + 2;
      textarea.setSelectionRange(newPosition, newPosition);
    }, 0);
  };

  const handleSubmit = async () => {
    if (!message.trim()) {
      setError('Please enter a message');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Replace variables in the message with actual values
      let processedMessage = message;
      if (data.sourceNodes && data.responses) {
        data.sourceNodes.forEach(nodeId => {
          const nodeResponse = data.responses[nodeId];
          if (nodeResponse) {
            processedMessage = processedMessage.replace(
              new RegExp(`\\{${nodeId}\\}`, 'g'),
              nodeResponse
            );
          }
        });
      }

      // Calculate prompt statistics
      const promptCount = promptHistory.filter(h => h.prompt === processedMessage).length;
      const lastResponse = promptHistory.length > 0 ? 
        promptHistory.filter(h => h.prompt === processedMessage).pop()?.response : null;
      
      // Update prompt history
      setPromptHistory(prev => [...prev, { 
        prompt: processedMessage,
        timestamp: new Date().toISOString()
      }]);

      const response = await fetch(`${process.env.REACT_APP_API_URL}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          node_id: id,
          message: processedMessage.trim(),
          system_instructions: `You are a focused AI assistant designed to provide increasingly optimized responses. Current context:
- This prompt has been used ${promptCount} times before
- Previous response length: ${lastResponse ? lastResponse.length : 'N/A'} characters
${promptCount > 0 ? '- This is a repeated prompt, indicating user wants maximum conciseness' : ''}

Follow these optimization guidelines:
1. If this is a repeated prompt, provide ONLY the essential output with no explanation
2. For mathematical or factual queries, return only the result
3. If processing data from other nodes, include only relevant portions
4. Format output in the most concise way possible while maintaining clarity
5. Remove ALL unnecessary words, context, or explanations
6. If the same prompt is used multiple times, it indicates the user wants the shortest possible valid response

Example optimization levels:
First time: "The sum of 10 + 10 is 20"
Second time: "20"
Third time onwards: 20

Your goal is to provide the most optimized response based on usage patterns.`
        }),
      });

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Failed to get response');
      }

      // Update history with the response
      setPromptHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1].response = result.response;
        return newHistory;
      });

      setResponse(result.response);
      if (data.onResponse) {
        data.onResponse(id, result.response);
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to get response from AI');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-node">
      <Handle 
        type="target" 
        position={Position.Top} 
        style={{ background: '#555' }}
      />
      <div className="node-content">
        <div className="node-header">
          <div className="node-info">
            <div className="node-id">ID: {id}</div>
            {promptHistory.length > 0 && (
              <div className="optimization-level" title="Optimization Level">
                <span className="optimization-icon">⚡</span>
                <span className="level-indicator">
                  {'.'.repeat(Math.min(promptHistory.filter(h => h.prompt === message).length, 3))}
                </span>
              </div>
            )}
          </div>
          <input
            type="text"
            value={nodeName}
            onChange={(e) => setNodeName(e.target.value)}
            className="node-name-input"
            placeholder="Enter node name..."
          />
        </div>
        {data.sourceNodes && data.sourceNodes.length > 0 && (
          <div className="connected-nodes">
            <label>Input from:</label>
            <div className="connection-list">
              {data.sourceNodes.map((nodeId) => (
                <div key={nodeId} className="connected-node">
                  {nodeId}
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="input-section">
          <div className="input-controls">
            <label>Input Prompt:</label>
            <div className="control-buttons">
              {data.sourceNodes && data.sourceNodes.length > 0 && (
                <button
                  type="button"
                  className="variable-button"
                  onClick={() => setShowVariableMenu(!showVariableMenu)}
                  disabled={isLoading}
                >
                  Insert Variable
                </button>
              )}
              {message.trim() && (
                <button
                  type="button"
                  className="clear-button"
                  onClick={() => setMessage('')}
                  disabled={isLoading}
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="input-container">
            <textarea
              ref={textAreaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your prompt for ChatGPT..."
              className="node-input"
              disabled={isLoading}
            />
            {showVariableMenu && data.sourceNodes && data.sourceNodes.length > 0 && (
              <div className="variable-menu" ref={variableMenuRef}>
                {data.sourceNodes.map((nodeId) => (
                  <button
                    key={nodeId}
                    onClick={() => insertVariable(nodeId)}
                    className="variable-option"
                  >
                    {nodeId}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <button 
          onClick={handleSubmit} 
          className={`submit-button ${isLoading ? 'loading' : ''}`}
          disabled={isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate Response'}
        </button>
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        <div className="output-section">
          <label>Output:</label>
          <div className="response-container">
            <textarea
              ref={responseAreaRef}
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder={isLoading ? 'Generating response...' : 'Response will appear here...'}
              className="node-input"
              style={{ minHeight: '100px' }}
              readOnly={isLoading}
            />
          </div>
        </div>
      </div>
      <Handle 
        type="source" 
        position={Position.Bottom} 
        style={{ background: '#555' }}
      />
    </div>
  );
};

export default ChatGPTNode;
