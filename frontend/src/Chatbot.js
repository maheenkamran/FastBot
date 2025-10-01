import { useState } from "react";
import axios from "axios";

function Chatbot() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");

    const sendMessage = async () => {
        if (!input) return;
        const res = await axios.post("http://127.0.0.1:8000/api/ask/", { query: input });
        setMessages([
            ...messages,
            { sender: "user", text: input },
            { sender: "bot", text: res.data.answer }
        ]);
        setInput("");
    };

    return (
        <div>
            <div>
                {messages.map((m, i) => (
                    <p key={i}><b>{m.sender}:</b> {m.text}</p>
                ))}
            </div>
            <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about admissions..."
            />
            <button onClick={sendMessage}>Send</button>
        </div>
    );
}

export default Chatbot;
