import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import AddCommentIcon from '@mui/icons-material/AddComment';
// import InfoIcon from '@mui/icons-material/Info';

const Navbar = () => {
  const handleNewChatClick = (e) => {
    e.preventDefault(); // Prevent the default anchor behavior
    window.location.href = "/"; // Reload the page
  };

  return (
    <div className="navbar">
      <Link to="/" onClick={handleNewChatClick} className="circle-link" data-hover="പുതിയ ചാറ്റ്">
        <AddCommentIcon style={{ fontSize: 45, color: '#007bff' }} />
      </Link>
      
    </div>
  );
};

export default Navbar;