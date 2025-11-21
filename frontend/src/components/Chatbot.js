import React, { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../services/api';
import './Chatbot.css';

const Chatbot = () => {
    const [messages, setMessages] = useState([
        { role: 'assistant', text: 'Hi! I\'m FastBot 🎓 Ask me anything about FAST-NUCES!' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
        setIsLoading(true);

        try {
            const response = await sendChatMessage(userMessage, conversationId);
            setConversationId(response.conversation_id);
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                text: response.answer,
                sources: response.sources 
            }]);
        } catch (error) {
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                text: 'Sorry, I encountered an error. Please try again.',
                isError: true 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const suggestedQuestions = [
        "What are the admission requirements?",
        "Tell me about fee structure",
        "What programs are offered?",
        "Hostel facilities available?"
    ];

    return (
        <div className="chatbot-container">
            {/* Header */}
            <div className="chatbot-header">
                <div className="header-content">
                    <div className="logo-section">
                        <div className="logo-icon">🤖</div>
                        <div className="logo-text">
                            <h1>FastBot</h1>
                            <span className="status">
                                <span className="status-dot"></span>
                                Online
                            </span>
                        </div>
                    </div>
                    <p className="tagline">Your FAST-NUCES Assistant</p>
                </div>
            </div>

            {/* Messages Area */}
            <div className="messages-container">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.role === 'assistant' && (
                            <div className="avatar assistant-avatar">🤖</div>
                        )}
                        <div className={`message-bubble ${msg.role} ${msg.isError ? 'error' : ''}`}>
                            <p>{msg.text}</p>
                            {msg.sources && msg.sources.length > 0 && msg.sources[0] !== 'FAQ' && (
                                <div className="sources">
                                    <span className="sources-label">📚 Sources:</span>
                                    {msg.sources.map((src, i) => (
                                        <span key={i} className="source-tag">{src}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                        {msg.role === 'user' && (
                            <div className="avatar user-avatar">👤</div>
                        )}
                    </div>
                ))}
                
                {isLoading && (
                    <div className="message assistant">
                        <div className="avatar assistant-avatar">🤖</div>
                        <div className="message-bubble assistant typing">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Suggested Questions */}
            {messages.length <= 1 && (
                <div className="suggestions">
                    <p className="suggestions-title">Try asking:</p>
                    <div className="suggestion-chips">
                        {suggestedQuestions.map((q, idx) => (
                            <button 
                                key={idx} 
                                className="suggestion-chip"
                                onClick={() => setInput(q)}
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Input Area */}
            <div className="input-container">
                <div className="input-wrapper">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask me about FAST-NUCES..."
                        disabled={isLoading}
                    />
                    <button 
                        onClick={handleSend} 
                        disabled={!input.trim() || isLoading}
                        className="send-button"
                    >
                        {isLoading ? (
                            <span className="loading-spinner"></span>
                        ) : (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" />
                            </svg>
                        )}
                    </button>
                </div>
                <p className="disclaimer">FastBot may make mistakes. Verify important information.</p>
            </div>
        </div>
    );
};

export default Chatbot;