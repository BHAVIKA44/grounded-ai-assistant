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
  const [adminStatus, setAdminStatus] = useState(null);
  const [adminModel, setAdminModel] = useState('llama3.2');
  const [adminProvider, setAdminProvider] = useState('ollama');
  const [supportedProviders, setSupportedProviders] = useState(['ollama', 'groq', 'openai']);
  const [supportedModels, setSupportedModels] = useState(['llama3.2', 'phi3:mini', 'qwen2.5:3b']);
  const [supportedModelsByProvider, setSupportedModelsByProvider] = useState({
    ollama: ['llama3.2', 'phi3:mini', 'qwen2.5:3b'],
    groq: ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768'],
    openai: [],
  });
  const [askTopK, setAskTopK] = useState(10);
  const [askUseRerank, setAskUseRerank] = useState(true);
  const [askUseCache, setAskUseCache] = useState(true);
  const [batchEvalInput, setBatchEvalInput] = useState('');
  const [batchEvalResult, setBatchEvalResult] = useState(null);
  const [batchEvalLoading, setBatchEvalLoading] = useState(false);
  const [ftJsonInput, setFtJsonInput] = useState('[{"question":"","context":"","answer":""}]');
  const [ftStatus, setFtStatus] = useState(null);
  const [adminLoading, setAdminLoading] = useState({
    status: false,
    model: false,
    cache: false,
    ftPrepare: false,
    ftRun: false,
    ftRefresh: false,
  });
  const [adminMessage, setAdminMessage] = useState({ type: '', text: '' });
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load documents on mount
  useEffect(() => {
    fetchDocuments();
    fetchAdminStatus();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/documents`);
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };
  const fetchAdminStatus = async () => {
    setAdminLoading((s) => ({ ...s, status: true }));
    try {
      const response = await axios.get(`${API_BASE}/admin/status`);
      setAdminStatus(response.data);
      if (response.data?.llm?.model) {
        setAdminModel(response.data.llm.model);
      }
      const llmRes = await axios.get(`${API_BASE}/admin/llm`);
      if (llmRes.data?.provider) {
        setAdminProvider(llmRes.data.provider);
      }
      if (Array.isArray(llmRes.data?.supported_providers)) {
        setSupportedProviders(llmRes.data.supported_providers);
      }
      if (Array.isArray(llmRes.data?.supported_models)) {
        setSupportedModels(llmRes.data.supported_models);
      }
      if (llmRes.data?.supported_models_by_provider) {
        setSupportedModelsByProvider(llmRes.data.supported_models_by_provider);
      }
      if (llmRes.data?.model) {
        setAdminModel(llmRes.data.model);
      }
      setAdminMessage({ type: 'success', text: 'Status refreshed.' });
    } catch (error) {
      console.error('Error fetching admin status:', error);
      setAdminMessage({ type: 'error', text: 'Failed to refresh status.' });
    } finally {
      setAdminLoading((s) => ({ ...s, status: false }));
    }
  };
  useEffect(() => {
    const nextModels = supportedModelsByProvider[adminProvider] || [];
    if (nextModels.length > 0) {
      setSupportedModels(nextModels);
      if (!nextModels.includes(adminModel)) {
        setAdminModel(nextModels[0]);
      }
    } else {
      setSupportedModels([]);
    }
  }, [adminProvider, supportedModelsByProvider]);

  const clearCache = async () => {
    setAdminLoading((s) => ({ ...s, cache: true }));
    try {
      await axios.delete(`${API_BASE}/admin/cache`, { params: { pattern: 'rag:*' } });
      await fetchAdminStatus();
      setAdminMessage({ type: 'success', text: 'Cache cleared successfully.' });
    } catch (error) {
      console.error('Error clearing cache:', error);
      setAdminMessage({ type: 'error', text: 'Failed to clear cache.' });
    } finally {
      setAdminLoading((s) => ({ ...s, cache: false }));
    }
  };
  const updateModel = async () => {
    setAdminLoading((s) => ({ ...s, model: true }));
    try {
      await axios.post(`${API_BASE}/admin/llm/pull`, null, { params: { model: adminModel } });
      await axios.post(`${API_BASE}/admin/llm`, null, { params: { provider: adminProvider, model: adminModel } });
      await fetchAdminStatus();
      setAdminMessage({ type: 'success', text: 'Model pulled and updated successfully.' });
    } catch (error) {
      console.error('Error updating model:', error);
      setAdminMessage({ type: 'error', text: 'Failed to pull/update model.' });
    } finally {
      setAdminLoading((s) => ({ ...s, model: false }));
    }
  };
  const runBatchEvaluation = async () => {
    setBatchEvalLoading(true);
    setBatchEvalResult(null);
    try {
      const payload = JSON.parse(batchEvalInput);
      const response = await axios.post(`${API_BASE}/evaluation/batch`, payload);
      setBatchEvalResult(response.data);
    } catch (error) {
      console.error('Error batch evaluating:', error);
      setBatchEvalResult(null);
    } finally {
      setBatchEvalLoading(false);
    }
  };
  const prepareFineTuning = async () => {
    setAdminLoading((s) => ({ ...s, ftPrepare: true }));
    try {
      const payload = JSON.parse(ftJsonInput);
      await axios.post(`${API_BASE}/admin/fine-tuning/prepare`, payload);
      const status = await axios.get(`${API_BASE}/admin/fine-tuning/status`);
      setFtStatus(status.data);
      setAdminMessage({ type: 'success', text: 'Fine-tuning dataset prepared.' });
    } catch (error) {
      console.error('Error preparing fine-tuning:', error);
      setAdminMessage({ type: 'error', text: 'Failed to prepare fine-tuning dataset.' });
    } finally {
      setAdminLoading((s) => ({ ...s, ftPrepare: false }));
    }
  };
  const runFineTuning = async () => {
    setAdminLoading((s) => ({ ...s, ftRun: true }));
    try {
      await axios.post(`${API_BASE}/admin/fine-tuning/run`);
      const status = await axios.get(`${API_BASE}/admin/fine-tuning/status`);
      setFtStatus(status.data);
      setAdminMessage({ type: 'success', text: 'Fine-tuning run finished.' });
    } catch (error) {
      console.error('Error running fine-tuning:', error);
      setAdminMessage({ type: 'error', text: 'Fine-tuning run failed.' });
    } finally {
      setAdminLoading((s) => ({ ...s, ftRun: false }));
    }
  };
  const refreshFineTuning = async () => {
    setAdminLoading((s) => ({ ...s, ftRefresh: true }));
    try {
      const status = await axios.get(`${API_BASE}/admin/fine-tuning/status`);
      setFtStatus(status.data);
      setAdminMessage({ type: 'success', text: 'Fine-tuning status refreshed.' });
    } catch (error) {
      console.error('Error refreshing fine-tuning status:', error);
      setAdminMessage({ type: 'error', text: 'Failed to refresh fine-tuning status.' });
    } finally {
      setAdminLoading((s) => ({ ...s, ftRefresh: false }));
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
        top_k: askTopK,
        use_reranking: askUseRerank,
        use_caching: askUseCache,
      });

      const answer = response.data.answer;
      const citations = Array.from(
        new Map((response.data.citations || []).map((c) => [c.source, c])).values()
      );

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
          <button
            className={`tab ${activeTab === 'admin' ? 'active' : ''}`}
            onClick={() => setActiveTab('admin')}
          >
            Admin
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
                      {typeof doc.file_size === 'number' && doc.file_size > 0 && (
                        <span className="file-size">
                          {formatFileSize(doc.file_size)}
                        </span>
                      )}
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
                            {cit.source} (score: {Math.max(0, Number(cit.score || 0)).toFixed(2)})
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
              <div className="ops-metrics" style={{ marginTop: '0.75rem' }}>
                <div className="ops-metric">
                  <span className="file-name">top_k</span>
                  <input className="input" type="number" min="1" max="50" value={askTopK} onChange={(e) => setAskTopK(Number(e.target.value || 10))} />
                </div>
                <div className="ops-metric">
                  <span className="file-name">rerank</span>
                  <input type="checkbox" checked={askUseRerank} onChange={(e) => setAskUseRerank(e.target.checked)} />
                </div>
                <div className="ops-metric">
                  <span className="file-name">cache</span>
                  <input type="checkbox" checked={askUseCache} onChange={(e) => setAskUseCache(e.target.checked)} />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'ops' && (
          <div className="card ops-card">
            <h2>RAG Operations</h2>

            <form onSubmit={runRetrievalTest} className="ops-section">
              <h3>Retrieval Inspector</h3>
              <input
                className="input"
                type="text"
                value={retrievalQuery}
                onChange={(e) => setRetrievalQuery(e.target.value)}
                placeholder="Enter a query to inspect retrieval quality"
              />
              <button className="btn btn-primary" type="submit" disabled={retrievalLoading}>
                {retrievalLoading ? 'Running...' : 'Run Retrieval'}
              </button>
            </form>

            {retrievalResults.length > 0 && (
              <div className="ops-list">
                {retrievalResults.map((item) => (
                  <div key={`${item.chunk_id}-${item.method}`} className="ops-item">
                    <div className="ops-item-head">
                      <span className="file-name">{item.source || 'unknown source'}</span>
                      <span className="file-size">
                        score {Number(item.score || 0).toFixed(3)} | {item.method}
                      </span>
                    </div>
                    <div className="ops-item-body">{item.content}</div>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={runEvaluation} className="ops-section">
              <h3>RAG Evaluation</h3>
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
              />
              <textarea
                className="input"
                rows={5}
                value={evalContexts}
                onChange={(e) => setEvalContexts(e.target.value)}
                placeholder="Retrieved contexts (one per line)"
              />
              <button className="btn btn-primary" type="submit" disabled={evalLoading}>
                {evalLoading ? 'Evaluating...' : 'Run Evaluation'}
              </button>
            </form>

            {evalResult && (
              <div className="ops-metrics">
                {Object.entries(evalResult).map(([metric, value]) => (
                  <div key={metric} className="ops-metric">
                    <span className="file-name">{metric}</span>
                    <span className="file-size">{Number(value).toFixed(4)}</span>
                  </div>
                ))}
              </div>
            )}
            <div className="ops-section">
              <h3>Batch Evaluation</h3>
              <textarea className="input" rows={6} value={batchEvalInput} onChange={(e) => setBatchEvalInput(e.target.value)} placeholder='[{"question":"...","ground_truth_answer":"...","retrieved_contexts":["..."]}]' />
              <button className="btn btn-primary" onClick={runBatchEvaluation} disabled={batchEvalLoading}>
                {batchEvalLoading ? 'Running...' : 'Run Batch Evaluation'}
              </button>
              {batchEvalResult && (
                <div className="ops-item">
                  <div className="ops-item-body">{JSON.stringify(batchEvalResult, null, 2)}</div>
                </div>
              )}
            </div>
          </div>
        )}
        {activeTab === 'admin' && (
          <div className="card ops-card">
            <h2>Admin Controls</h2>
            {adminMessage.text && (
              <div className={adminMessage.type === 'error' ? 'error' : 'loading'}>
                {adminMessage.text}
              </div>
            )}
            <div className="ops-section">
              <h3>System Status</h3>
              <button className="btn btn-primary" onClick={fetchAdminStatus} disabled={adminLoading.status}>
                {adminLoading.status ? 'Refreshing...' : 'Refresh Status'}
              </button>
              {adminStatus && (
                <div className="ops-metrics">
                  <div className="ops-metric"><span className="file-name">DB</span><span className="file-size">{adminStatus.services?.db ? 'up' : 'down'}</span></div>
                  <div className="ops-metric"><span className="file-name">Redis</span><span className="file-size">{adminStatus.services?.redis ? 'up' : 'down'}</span></div>
                  <div className="ops-metric"><span className="file-name">LangSmith</span><span className="file-size">{adminStatus.services?.langsmith ? 'enabled' : 'disabled'}</span></div>
                </div>
              )}
            </div>
            <div className="ops-section">
              <h3>LLM Configuration</h3>
              <select className="input" value={adminProvider} onChange={(e) => setAdminProvider(e.target.value)}>
                {supportedProviders.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              {supportedModels.length > 0 ? (
                <select className="input" value={adminModel} onChange={(e) => setAdminModel(e.target.value)}>
                  {supportedModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              ) : (
                <input className="input" value={adminModel} onChange={(e) => setAdminModel(e.target.value)} placeholder="model name" />
              )}
              <div className="chat-input-container">
                <button className="btn btn-primary" onClick={updateModel} disabled={adminLoading.model}>
                  {adminLoading.model ? 'Updating...' : 'Update Model'}
                </button>
              </div>
            </div>
            <div className="ops-section">
              <h3>Cache Control</h3>
              <button className="btn btn-danger" onClick={clearCache} disabled={adminLoading.cache}>
                {adminLoading.cache ? 'Clearing...' : 'Clear Cache'}
              </button>
            </div>
            <div className="ops-section">
              <h3>Fine-tuning</h3>
              <textarea className="input" rows={6} value={ftJsonInput} onChange={(e) => setFtJsonInput(e.target.value)} placeholder='[{"question":"...","context":"...","answer":"..."}]' />
              <div className="chat-input-container">
                <button className="btn btn-primary" onClick={prepareFineTuning} disabled={adminLoading.ftPrepare}>
                  {adminLoading.ftPrepare ? 'Preparing...' : 'Prepare Dataset'}
                </button>
                <button className="btn btn-primary" onClick={runFineTuning} disabled={adminLoading.ftRun}>
                  {adminLoading.ftRun ? 'Running...' : 'Run Fine-tuning'}
                </button>
                <button className="btn" onClick={refreshFineTuning} disabled={adminLoading.ftRefresh}>
                  {adminLoading.ftRefresh ? 'Refreshing...' : 'Refresh Status'}
                </button>
              </div>
              {ftStatus && (
                <div className="ops-item">
                  <div className="ops-item-body">{JSON.stringify(ftStatus, null, 2)}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
