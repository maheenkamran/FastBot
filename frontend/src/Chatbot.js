import { useState } from "react";
import axios from "axios";
import './Chatbot.css';

function Chatbot() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [listening, setListening] = useState(false);

    // Text-to-speech function
    const speak = (text) => {
        if (!window.speechSynthesis) return;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        utterance.pitch = 1;
        utterance.rate = 1.4;
        utterance.volume = 1;
        window.speechSynthesis.speak(utterance);
    };

    // Function to send message to backend
    const sendMessage = async (text, shouldSpeak = false) => {
        if (!text) return;

        try {
            const res = await axios.post("http://127.0.0.1:8000/api/ask/", { question: text });
            const botReply = res.data.answer;

            setMessages((prev) => [
                ...prev,
                { sender: "user", text: text },
                { sender: "bot", text: botReply }
            ]);
            setInput("");

            if (shouldSpeak) speak(botReply); // only speak if flag is true (voice input)

        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => [
                ...prev,
                { sender: "user", text: text },
                { sender: "bot", text: "Sorry, the server is not responding." }
            ]);
        }
    };

    // Voice input handler
    const handleVoice = () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Your browser does not support speech recognition");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = "en-US";

        setListening(true);
        recognition.start();

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            setInput(transcript);
            sendMessage(transcript, true); // voice input: speak automatically
        };

        recognition.onend = () => setListening(false);
        recognition.onerror = () => setListening(false);
    };

    return (
        <div>
            <div>
                {messages.map((m, i) => (
                    <p key={i} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                        <b>{m.sender}:</b> {m.text}
                        {m.sender === "bot" && (
                            <span
                                style={{ cursor: "pointer", fontSize: "16px" }}
                                onClick={() => speak(m.text)}
                                title="Listen"
                            >
                                🔊
                            </span>
                        )}
                    </p>
                ))}
            </div>

            <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === "Enter") sendMessage(input, false); // typed messages: no auto speak
                }}
                placeholder="Ask about admissions..."
            />

            <button onClick={() => sendMessage(input, false)}>Send</button>

            <button onClick={handleVoice}>
                🎤 Talk
                {listening && <span className="listening-indicator"></span>}
            </button>
        </div>
    );
}

export default Chatbot;
