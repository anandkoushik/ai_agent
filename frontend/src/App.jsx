import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Paperclip, X, Bot, User, Activity, AlertCircle, Archive, Database, Cpu, UploadCloud, Layers } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [datasetFile, setDatasetFile] = useState(null);
  const [inferenceFile, setInferenceFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const ws = useRef(null);
  const messagesEndRef = useRef(null);
  const datasetInputRef = useRef(null);
  const inferenceInputRef = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/chat');
    ws.current.onopen = () => {
      console.log('WebSocket Connected');
      setMessages([{ sender: 'system', text: 'Connected to AI Training Orchestrator via WebSocket.' }]);
    };
    ws.current.onmessage = (event) => {
      setMessages(prev => [...prev, { sender: 'system', text: event.data }]);
    };
    ws.current.onerror = (error) => console.error('WebSocket Error:', error);
    return () => { if (ws.current) ws.current.close(); };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const abortControllerRef = useRef(null);

  const executeWorkflow = async (overridePrompt = null) => {
    const activePrompt = overridePrompt || input;
    if (!activePrompt.trim() && !datasetFile && !inferenceFile) return;

    setLoading(true);
    setMessages(prev => [...prev, { sender: 'user', text: overridePrompt ? `[System: Triggered ${overridePrompt}]` : activePrompt }]);
    
    abortControllerRef.current = new AbortController();
    
    const formData = new FormData();
    formData.append('prompt', activePrompt || 'execute workflow');
    formData.append('session_id', 'user-session-123');
    if (datasetFile) formData.append('dataset', datasetFile);
    if (inferenceFile) formData.append('inference_target', inferenceFile);

    if (!overridePrompt) setInput('');
    
    try {
      const response = await fetch('http://localhost:8000/chat/workflow', {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal,
      });
      const data = await response.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { sender: 'bot', text: `Error: ${data.error}` }]);
      } else if (['conversation', 'general_ml_question', 'troubleshooting', 'multi_intent', 'concept_explanation', 'hyperparameter_guidance'].includes(data.intent)) {
        setMessages(prev => [...prev, { sender: 'bot', text: data.response }]);
      } else {
        const wf = data.workflow_response;
        if (wf?.status === 'error') {
          setMessages(prev => [...prev, { sender: 'bot', text: `Workflow Error: ${wf.message}` }]);
        } else if (wf) {
          const msg = `Workflow [${wf.workflow_id}] Completed!\nModality: ${wf.manifest.dataset_type}\nModel: ${wf.manifest.recommended_model}\nEpochs Selected: ${wf.hyperparameters?.estimated_epochs || 'N/A'}\nTraining Status: ${wf.training_result?.status || 'N/A'}`;
          setMessages(prev => [...prev, { sender: 'bot', text: msg }]);
          if (data.hybrid_explanation) {
              setMessages(prev => [...prev, { sender: 'bot', text: `Explanation: ${data.hybrid_explanation}` }]);
          }
        } else {
           setMessages(prev => [...prev, { sender: 'bot', text: data.response || "Task completed." }]);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') return;
      console.error('Error:', error);
      setMessages(prev => [...prev, { sender: 'bot', text: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setLoading(false);
      setDatasetFile(null);
      setInferenceFile(null);
      if (datasetInputRef.current) datasetInputRef.current.value = '';
      if (inferenceInputRef.current) inferenceInputRef.current.value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    executeWorkflow();
  };

  const cancelWorkflow = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setLoading(false);
    setMessages(prev => [...prev, { sender: 'system', text: 'Workflow cancelled by user.' }]);
  };

  const handleFileSelect = (type) => {
    if (type === 'dataset' && datasetInputRef.current) datasetInputRef.current.click();
    if (type === 'inference' && inferenceInputRef.current) inferenceInputRef.current.click();
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#050810] text-gray-100 font-sans relative">
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-purple-600/20 blur-[120px]"></div>
        <div className="absolute top-[60%] -right-[10%] w-[40%] h-[60%] rounded-full bg-blue-600/20 blur-[120px]"></div>
      </div>

      <header className="relative z-10 glass-panel border-b border-white/5 px-6 py-4 flex justify-between items-center rounded-b-2xl mx-2 mt-2">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg shadow-lg shadow-purple-500/30">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold tracking-tight glow-text">Antigravity AI</h1>
        </div>
        <div className="flex items-center space-x-2 bg-white/5 px-3 py-1.5 rounded-full border border-white/5">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </span>
          <span className="text-sm font-medium text-emerald-400">Online</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative z-10 w-full max-w-[1400px] mx-auto px-2 pb-2 mt-2 gap-4">
        
        {/* Main Chat Column */}
        <div className="flex flex-col flex-1 h-full min-w-0 glass-panel rounded-2xl border border-white/5 overflow-hidden shadow-2xl">
          <main className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scroll-smooth">
            <AnimatePresence>
              {messages.map((msg, idx) => (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 15, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex max-w-[85%] sm:max-w-2xl gap-3 ${msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-lg ${
                      msg.sender === 'user' ? 'bg-blue-600 shadow-blue-500/30' : msg.sender === 'system' ? 'bg-gray-700 shadow-gray-900/30' : 'bg-purple-600 shadow-purple-500/30'
                    }`}>
                      {msg.sender === 'user' ? <User className="w-4 h-4 text-white" /> : msg.sender === 'system' ? <AlertCircle className="w-4 h-4 text-gray-300" /> : <Bot className="w-4 h-4 text-white" />}
                    </div>
                    <div className={`rounded-2xl p-4 shadow-xl ${
                      msg.sender === 'user' ? 'bg-blue-600/90 text-white rounded-tr-sm border border-blue-500/50' : msg.sender === 'system' ? 'bg-black/40 border border-white/5 text-gray-400 rounded-tl-sm text-sm font-mono' : 'bg-white/5 border border-white/10 text-gray-200 rounded-tl-sm'
                    }`}>
                      {msg.sender === 'system' && (
                        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block mb-1">System Log</span>
                      )}
                      <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.text}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
              {loading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex justify-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
                     <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-2xl rounded-tl-sm p-4 flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-gray-400 ml-2 italic">Processing workflow...</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} className="h-4" />
          </main>

          <footer className="p-4 border-t border-white/5 bg-black/20">
            <form onSubmit={handleSubmit} className="flex flex-col space-y-3">
              <input type="file" ref={datasetInputRef} onChange={(e) => setDatasetFile(e.target.files[0])} className="hidden" />
              <input type="file" ref={inferenceInputRef} onChange={(e) => setInferenceFile(e.target.files[0])} className="hidden" />
              
              <div className="relative flex items-center bg-black/40 border border-white/10 rounded-2xl p-1 pr-2 group hover:border-purple-500/50 transition-colors">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question..."
                  className="flex-1 bg-transparent border-none text-gray-100 px-4 py-3 focus:outline-none focus:ring-0 placeholder-gray-500 text-lg"
                />
                
                <div className="flex space-x-2">
                   {loading ? (
                    <button type="button" onClick={cancelWorkflow} className="bg-red-500/20 text-red-400 p-3 rounded-xl hover:bg-red-500/30 transition-all border border-red-500/30">
                      <X className="w-5 h-5" />
                    </button>
                   ) : (
                    <button type="submit" disabled={!input.trim()} className="bg-gradient-to-r from-blue-600 to-purple-600 p-3 rounded-xl text-white font-semibold hover:shadow-lg hover:shadow-purple-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all">
                      <Send className="w-5 h-5" />
                    </button>
                   )}
                </div>
              </div>
            </form>
          </footer>
        </div>

        {/* Sidebar / Control Panel */}
        <aside className="hidden lg:flex flex-col w-[350px] flex-shrink-0 h-full space-y-4">
          
          {/* Dataset & Training Card */}
          <div className="glass-panel border border-white/5 rounded-2xl p-5 flex flex-col shadow-2xl flex-1">
            <div className="flex items-center space-x-2 mb-4 text-purple-400">
              <Database className="w-5 h-5" />
              <h2 className="font-semibold text-lg text-white">Dataset & Training</h2>
            </div>
            
            <p className="text-sm text-gray-400 mb-4">Upload a ZIP file containing your dataset (YOLO, Whisper, or Text format) to begin automated training.</p>
            
            <div 
              onClick={() => handleFileSelect('dataset')}
              className={`flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all mb-4
                ${datasetFile ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 hover:border-purple-400 hover:bg-white/5'}`}
            >
              {datasetFile ? (
                <>
                  <Archive className="w-10 h-10 text-purple-400 mb-3" />
                  <span className="text-sm text-gray-200 font-medium truncate w-full px-4">{datasetFile.name}</span>
                  <span className="text-xs text-gray-500 mt-1">{(datasetFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-10 h-10 text-gray-500 mb-3" />
                  <span className="text-sm text-gray-300 font-medium">Click to upload dataset</span>
                  <span className="text-xs text-gray-500 mt-1">.zip format supported</span>
                </>
              )}
            </div>

            <button 
              onClick={() => executeWorkflow('train')}
              disabled={!datasetFile || loading}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-semibold py-3 px-4 rounded-xl shadow-lg shadow-purple-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
            >
              <Cpu className="w-4 h-4" />
              <span>Train Model</span>
            </button>
          </div>

          {/* Inference Card */}
          <div className="glass-panel border border-white/5 rounded-2xl p-5 flex flex-col shadow-2xl flex-1">
            <div className="flex items-center space-x-2 mb-4 text-blue-400">
              <Layers className="w-5 h-5" />
              <h2 className="font-semibold text-lg text-white">Inference</h2>
            </div>
            
            <p className="text-sm text-gray-400 mb-4">Test your trained model by uploading a target image, audio file, or text prompt.</p>

            <div 
              onClick={() => handleFileSelect('inference')}
              className={`flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all mb-4
                ${inferenceFile ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-blue-400 hover:bg-white/5'}`}
            >
              {inferenceFile ? (
                <>
                  <Archive className="w-10 h-10 text-blue-400 mb-3" />
                  <span className="text-sm text-gray-200 font-medium truncate w-full px-4">{inferenceFile.name}</span>
                  <span className="text-xs text-gray-500 mt-1">{(inferenceFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-10 h-10 text-gray-500 mb-3" />
                  <span className="text-sm text-gray-300 font-medium">Click to upload target</span>
                  <span className="text-xs text-gray-500 mt-1">Image, Audio, or Text</span>
                </>
              )}
            </div>

            <button 
              onClick={() => executeWorkflow('run inference')}
              disabled={!inferenceFile || loading}
              className="w-full bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-semibold py-3 px-4 rounded-xl shadow-lg shadow-blue-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
            >
              <Activity className="w-4 h-4" />
              <span>Run Inference</span>
            </button>
          </div>
          
        </aside>
      </div>
    </div>
  );
}

export default App;
