import React, { useState } from 'react';
import { sendChatMessage } from '../services/api';

function Chatbot() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [conversationId, setConversationId] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (!input.trim()) return;

        // Add user message to chat
        const userMessage = { role: 'user', text: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            // Send to backend
            const response = await sendChatMessage(input, conversationId);
            
            // Store conversation ID for follow-up messages
            if (!conversationId) {
                setConversationId(response.conversation_id);
            }

            // Add assistant message to chat
            const assistantMessage = {
                role: 'assistant',
                text: response.answer,
                usedFaq: response.used_faq,
                sources: response.sources
            };
            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            const errorMessage = {
                role: 'assistant',
                text: 'Sorry, something went wrong. Please try again.',
                error: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
            <h1>FAST-NUCES Chatbot</h1>
            
            {/* Chat Messages */}
            <div style={{
                border: '1px solid #ccc',
                borderRadius: '8px',
                padding: '20px',
                height: '500px',
                overflowY: 'auto',
                marginBottom: '20px',
                backgroundColor: '#f9f9f9'
            }}>
                {messages.map((msg, index) => (
                    <div key={index} style={{
                        marginBottom: '15px',
                        textAlign: msg.role === 'user' ? 'right' : 'left'
                    }}>
                        <div style={{
                            display: 'inline-block',
                            padding: '10px 15px',
                            borderRadius: '8px',
                            backgroundColor: msg.role === 'user' ? '#007bff' : '#e9ecef',
                            color: msg.role === 'user' ? 'white' : 'black',
                            maxWidth: '70%',
                            textAlign: 'left'
                        }}>
                            <strong>{msg.role === 'user' ? 'You' : 'Bot'}:</strong>
                            <p style={{ margin: '5px 0', whiteSpace: 'pre-wrap' }}>
                                {msg.text}
                            </p>
                            {msg.sources && msg.sources.length > 0 && (
                                <small style={{ 
                                    display: 'block', 
                                    marginTop: '5px',
                                    opacity: 0.8,
                                    fontStyle: 'italic'
                                }}>
                                    {msg.usedFaq ? '📚 From FAQ' : `📄 ${msg.sources.join(', ')}`}
                                </small>
                            )}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div style={{ textAlign: 'center', color: '#999' }}>
                        Bot is typing...
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div style={{ display: 'flex', gap: '10px' }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask a question about FAST-NUCES..."
                    style={{
                        flex: 1,
                        padding: '12px',
                        fontSize: '16px',
                        borderRadius: '4px',
                        border: '1px solid #ccc'
                    }}
                    disabled={loading}
                />
                <button
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    style={{
                        padding: '12px 24px',
                        fontSize: '16px',
                        borderRadius: '4px',
                        backgroundColor: '#007bff',
                        color: 'white',
                        border: 'none',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1
                    }}
                >
                    Send
                </button>
            </div>
        </div>
    );
}

export default Chatbot;