import { FormEvent, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";

interface ChatbotPanelProps {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (message: string) => Promise<void> | void;
}

export function ChatbotPanel({ messages, loading, onSend }: ChatbotPanelProps) {
  const [input, setInput] = useState("");
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim()) {
      return;
    }
    await onSend(input.trim());
    setInput("");
  };

  return (
    <section className="chatbot">
      <header>
        <h2>Trading copilot</h2>
        <span className="hint">Ask strategy questions, get market colour, or automate simulations</span>
      </header>

      <div className="chat-window" ref={containerRef}>
        {messages.length === 0 ? (
          <p className="empty">Say hi or ask for a quick strategy idea. The assistant can also spin up a simulation for you.</p>
        ) : (
          messages.map((message, index) => (
            <article key={`${message.timestamp}-${index}`} className={`bubble ${message.role}`}>
              <div className="content">{message.content}</div>
              {message.citations?.length ? (
                <ul className="citations">
                  {message.citations.map((citation) => (
                    <li key={citation}>
                      <a href={citation} target="_blank" rel="noreferrer">{citation}</a>
                    </li>
                  ))}
                </ul>
              ) : null}
              {message.actions?.length ? (
                <ul className="actions">
                  {message.actions.map((action, actionIndex) => (
                    <li key={`${action.type}-${actionIndex}`}>
                      <strong>{action.label}</strong>
                    </li>
                  ))}
                </ul>
              ) : null}
            </article>
          ))
        )}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={loading ? "Waiting for assistant..." : "Ask about a symbol, strategy, or create a simulation"}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  );
}
