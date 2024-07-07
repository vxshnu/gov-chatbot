import React, { useEffect, useRef, useState } from 'react';
import './ChatBox.css';
import axios from 'axios';
import ThumbsUpDownIcon from '@mui/icons-material/ThumbsUpDown';

const ChatBox = ({ messages, addFeedback }) => {
  const chatContainerRef = useRef(null);
  const [feedbackIndex, setFeedbackIndex] = useState(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [showPopup, setShowPopup] = useState(false);

  useEffect(() => {
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [messages]);

  const toggleFeedbackInput = (index) => {
    setFeedbackIndex(feedbackIndex === index ? null : index);
    setFeedbackText(''); // Clear the feedback input text when toggling
  };

  const handleFeedbackChange = (event) => {
    setFeedbackText(event.target.value);
  };

  const handleKeyPress = (index, event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      const { value } = event.target;
      addFeedback(index, value); // Save feedback when Enter is pressed
      sendFeedbackToServer(index, value); // Send feedback to the server
      setFeedbackText(''); // Clear the feedback input text
      setFeedbackIndex(null); // Close the feedback input
    }
  };

  const sendFeedbackToServer = async (index, feedback) => {
    try {
      const response = await axios.post('http://localhost:5000/feedback', {
        feedback,
        index
      });
      console.log('Feedback response:', response.data);
      // Optionally, you can add logic here to update the UI with the feedback response
      setShowPopup(true); // Show the success popup
      setTimeout(() => {
        setShowPopup(false); // Hide the popup after 3 seconds
      }, 3000);
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  };

  return (
    <div ref={chatContainerRef} className="chat-box">
      {messages.map((msg, index) => (
        <div key={index} className={`chat-message ${msg.sender}`}>
          <div className={`message ${msg.sender}`}>{msg.text}</div>
          {msg.sender === 'bot' && (
            <div className="feedback-container">
              {feedbackIndex === index ? (
                <div className="feedback-input">
                  <input
                    type="text"
                    placeholder="Provide feedback..."
                    value={feedbackText}
                    onChange={handleFeedbackChange}
                    onKeyPress={(e) => handleKeyPress(index, e)}
                  />
                </div>
              ) : (
                <button className="feedback-button" onClick={() => toggleFeedbackInput(index)}>
                  <ThumbsUpDownIcon/>
                </button>
              )}
            </div>
          )}
        </div>
      ))}
      {showPopup && (
        <div className="popup">
          നിങ്ങളുടെ അഭിപ്രായത്തിന് നന്ദി!
        </div>
      )}
    </div>
  );
};

export default ChatBox;