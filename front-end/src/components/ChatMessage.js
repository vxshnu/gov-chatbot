import React from 'react';
import './ChatMessage.css';

function ChatMessage({ message }) {
  const { text, sender } = message;
  const messageClass = sender === 'user' ? 'chat-message user' : 'chat-message bot';
  
  return (
    <div className={messageClass}>
      <div className="message-content">{text}</div>
    </div>
  );
}

export default ChatMessage;