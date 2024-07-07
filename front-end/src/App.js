import React, { useState } from 'react';
import './App.css';
import ChatBox from './components/ChatBox';
import ChatInput from './components/ChatInput';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import AboutUs from './components/AboutUs';
import { Link } from 'react-router-dom';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [isVisible, setIsVisible] = useState(true);

  const addMessage = (message, sender) => {
    setMessages(prevMessages => [...prevMessages, { text: message, sender, feedback: '' }]);
  };
  const addFeedback = (index, feedback) => {
    setMessages(prevMessages =>
      prevMessages.map((msg, i) =>
        i === index ? { ...msg, feedback } : msg
      )
    );
  };

  const handleSendClick = () => {
    setIsVisible(false);
  };

  return (
    <div className="main-content">
      
      <div className='header'>
        <h1>ജന സഹായി</h1>
        <Link to="/aboutus" className="aboutus circle-link" data-hover="About Us">
        {}
        ABOUT US
        </Link>
      </div>
      <div className="chat-container">
        {isVisible && <h2>എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാനാകും?</h2>}
        <ChatBox messages={messages} addFeedback={addFeedback} />
        <ChatInput addMessage={addMessage} onSend={handleSendClick} />
        {}
        <Navbar />
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/aboutus" element={<AboutUs />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
