import React, { useState } from 'react';
import './ChatInput.css';
import axios from 'axios';

const ChatInput = ({ addMessage, onSend }) => {
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);

  const handleSend = async () => {
    if (input.trim()) {
      addMessage(input, 'user');
      setIsSending(true);
      const messageToSend = input;
      setInput('');

      try {
        if (messageToSend.trim().toLowerCase() === 'അതെ') {
          const response = await axios.get('http://localhost:5000/get_certificate', {
            responseType: 'blob' // Set responseType to 'blob' to receive binary data
          });

          if (!response.data) {
            throw new Error('Empty response received');
          }

          // Create a blob URL for the received data
          const blob = new Blob([response.data]);
          const url = window.URL.createObjectURL(blob);

          // Create a hidden anchor element
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', 'doc.pdf'); // Set desired filename here
          link.style.display = 'none'; // Hide the anchor element
          document.body.appendChild(link);

          // Programmatically click the anchor to trigger download
          link.click();

          // Clean up: remove the anchor element and revoke the blob URL
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);

          // Optionally, inform the user about the download action
          addMessage('എല്ലാ രേഖകളും ഡൗൺലോഡ് ചെയ്യുന്നു', 'bot');
        } else {
          // Handle other messages or call your backend as needed
          const response = await fetch('http://127.0.0.1:5000/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: messageToSend })
          });

          if (!response.ok) {
            throw new Error('Network response was not ok');
          }

          const data = await response.json();
          addMessage(data.answer, 'bot');
        }
      } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
      }

      setIsSending(false);
      onSend(); // Call the onSend function when the message is sent
    }
  };

  return (
    <div className="chat-input">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && !isSending && handleSend()}
        placeholder="ഒരു സന്ദേശം ടൈപ്പ് ചെയ്യുക..."
      />
      <button onClick={handleSend} disabled={isSending} aria-label="Send message">
        {isSending ? 'Sending...' : 'അയയ്ക്കുക'}
      </button>
    </div>
  );
};

export default ChatInput;
