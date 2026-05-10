import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './index.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [retrievalQuery, setRetrievalQuery] = useState('');
  const [retrievalResults, setRetrievalResults] = useState([]);
  const [retrievalLoading, setRetrievalLoading] = useState(false);
  const [evalQuestion, setEvalQuestion] = useState('');
  const [evalGroundTruth, setEvalGroundTruth] = useState('');
  const [evalContexts, setEvalContexts] = useState('');
  const [evalResult, setEvalResult] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/documents`);
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      await uploadFiles(files);
    }
  };

  const handleFileSelect = async (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await uploadFiles(files);
    }
  };

  const uploadFiles = async (files) => {
    setUploading(true);
    try {
      for (let i = 0; i < files.length; i++) {
        const formData = new FormData();
        formData.append('file', files[i]);
        await axios.post(`${API_BASE}/documents/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }
      await fetchDocuments();
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Failed to upload files. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      await axios.delete(`${API_BASE}/documents/${docId}`);
      await fetchDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const runRetrievalTest = async (e) => {
    e.preventDefault();
    if (!retrievalQuery.trim()) return;
    setRetrievalLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/retrieval`, {
        params: { query: retrievalQuery.trim(), top_k: 10 },
      });
      setRetrievalResults(response.data.results || []);
    } catch (error) {
      console.error('Error testing retrieval:', error);
      setRetrievalResults([]);
    } finally {
      setRetrievalLoading(false);
    }
  };

  const runEvaluation = async (e) => {
    e.preventDefault();
    if (!evalQuestion.trim() || !evalGroundTruth.trim() || !evalContexts.trim()) return;
    const contexts = evalContexts
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    setEvalLoading(true);
    setEvalResult(null);
    try {
      const response = await axios.post(`${API_BASE}/evaluation`, {
        question: evalQuestion.trim(),
        ground_truth_answer: evalGroundTruth.trim(),
        retrieved_contexts: contexts,
      });
      setEvalResult(response.data);
    } catch (error) {
      console.error('Error running evaluation:', error);
      setEvalResult(null);
    } finally {
      setEvalLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);

    try {
      const response = await axios.post(`${API_BASE}/ask`, {
        question: userMessage,
        include_citations: true,
      });

      const answer = response.data.answer;
      const citations = response.data.citations || [];

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: answer,
          citations,
          metrics: {
            retrieval: response.data.retrieval_time_ms,
            generation: response.data.generation_time_ms,
            total: response.data.total_time_ms,
            cached: response.data.cached,
            model: response.data.model_used,
          },
        },
      ]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request.',
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Grounded AI Assistant</h1>
      </header>

      <main className="main-content">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button
            className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Documents ({documents.length})
          </button>
          <button
            className={`tab ${activeTab === 'ops' ? 'active' : ''}`}
            onClick={() => setActiveTab('ops')}
          >
            RAG Ops
          </button>
        </div>

        {activeTab === 'upload' && (
          <div className="card">
            <h2>Upload Documents</h2>
            <div
              className={`upload-area ${dragActive ? 'dragging' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon">📄</div>
              <p>Drag and drop files here, or click to select</p>
              <p style={{ color: '#666', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Supported formats: PDF, TXT, DOCX
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.docx"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </div>

            {uploading && (
              <div className="loading">
                <div className="spinner"></div>
                Uploading...
              </div>
            )}

            {documents.length > 0 && (
              <div className="file-list">
                <h3 style={{ marginBottom: '0.5rem' }}>Uploaded Documents</h3>
                {documents.map((doc) => (
                  <div key={doc.id} className="file-item">
                    <div className="file-info">
                      <span className="file-name">{doc.title}</span>
                      <span className="file-size">
                        {formatFileSize(doc.file_size || 0)}
                      </span>
                    </div>
                    <button
                      className="btn btn-danger"
                      onClick={() => deleteDocument(doc.id)}
                    >
                      Delete
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="card">
            <h2>Ask Questions</h2>
            <div className="chat-container">
              <div className="messages">
                {messages.length === 0 && (
                  <div style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
                    <p>No messages yet. Start a conversation!</p>
                    <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                      Upload documents first to enable RAG-powered answers.
                    </p>
                  </div>
                )}
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations">
                        <strong>Sources:</strong>
                        {msg.citations.map((cit, citIdx) => (
                          <span key={citIdx} className="citation">
                            {cit.source} (score: {cit.score?.toFixed(2)})
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.metrics && (
                      <div className="citations">
                        <strong>Performance:</strong>
                        <span className="citation">
                          retrieval {msg.metrics.retrieval?.toFixed?.(1)} ms
                        </span>
                        <span className="citation">
                          generation {msg.metrics.generation?.toFixed?.(1)} ms
                        </span>
                        <span className="citation">
                          total {msg.metrics.total?.toFixed?.(1)} ms
                        </span>
                        <span className="citation">
                          {msg.metrics.cached ? 'cached' : 'fresh'}
                        </span>
                        <span className="citation">{msg.metrics.model}</span>
                      </div>
                    )}
                  </div>
                ))}
                {loading && (
                  <div className="message assistant">
                    <div className="loading">
                      <div className="spinner"></div>
                      Thinking...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <form className="chat-input-container" onSubmit={handleSubmit}>
                <textarea
                  className="chat-input"
                  placeholder="Ask a question about your documents..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  rows={1}
                  disabled={loading}
                />
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !input.trim()}
                >
                  Send
                </button>
              </form>
            </div>
          </div>
        )}

        {activeTab === 'ops' && (
          <div className="card">
            <h2>RAG Operations</h2>

            <form onSubmit={runRetrievalTest} style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '0.5rem' }}>Retrieval Inspector</h3>
              <input
                className="input"
                type="text"
                value={retrievalQuery}
                onChange={(e) => setRetrievalQuery(e.target.value)}
                placeholder="Enter a query to inspect retrieval quality"
              />
              <button className="btn" type="submit" disabled={retrievalLoading}>
                {retrievalLoading ? 'Running...' : 'Run Retrieval'}
              </button>
            </form>

            {retrievalResults.length > 0 && (
              <div className="file-list" style={{ marginBottom: '1.5rem' }}>
                {retrievalResults.map((item) => (
                  <div key={`${item.chunk_id}-${item.method}`} className="file-item">
                    <div className="file-info">
                      <span className="file-name">{item.source}</span>
                      <span className="file-size">
                        score {Number(item.score || 0).toFixed(3)} | {item.method}
                      </span>
                      <span style={{ color: '#555', marginTop: 6 }}>{item.content}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={runEvaluation}>
              <h3 style={{ marginBottom: '0.5rem' }}>RAG Evaluation</h3>
              <input
                className="input"
                type="text"
                value={evalQuestion}
                onChange={(e) => setEvalQuestion(e.target.value)}
                placeholder="Evaluation question"
              />
              <textarea
                className="input"
                rows={3}
                value={evalGroundTruth}
                onChange={(e) => setEvalGroundTruth(e.target.value)}
                placeholder="Ground truth answer"
                style={{ marginTop: '0.5rem' }}
              />
              <textarea
                className="input"
                rows={5}
                value={evalContexts}
                onChange={(e) => setEvalContexts(e.target.value)}
                placeholder="Retrieved contexts (one per line)"
                style={{ marginTop: '0.5rem' }}
              />
              <button className="btn" type="submit" disabled={evalLoading}>
                {evalLoading ? 'Evaluating...' : 'Run Evaluation'}
              </button>
            </form>

            {evalResult && (
              <div className="file-list" style={{ marginTop: '1rem' }}>
                {Object.entries(evalResult).map(([metric, value]) => (
                  <div key={metric} className="file-item">
                    <div className="file-info">
                      <span className="file-name">{metric}</span>
                      <span className="file-size">{Number(value).toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
