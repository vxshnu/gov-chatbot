import React from 'react';
import './AboutUs.css';

const AboutUs = () => {
  return (
    <div className="aboutus-page">
      <h1>About Us</h1>
      <p>Welcome to Jana Sahayi!</p>

      <section>
        <h2>Our Mission</h2>
        <p>
          At Jana Sahayi, our mission is to provide reliable and efficient solutions to assist you with your needs. We aim to leverage technology to make your life easier and more productive.
        </p>
      </section>

      <section>
        <h2>Our Team</h2>
        <p>
          Our team is composed of dedicated professionals who are passionate about technology and innovation. We work hard to ensure that our application meets your expectations and provides a seamless user experience.
        </p>
      </section>

      <section>
        <h2>Our History</h2>
        <p>
          Jana Sahayi was founded in [Year] with the vision to create a platform that bridges the gap between technology and user needs. Over the years, we have evolved and adapted to the latest trends and technologies to deliver the best possible service.
        </p>
      </section>

      <section>
        <h2>Contact Us</h2>
        <p>
          We'd love to hear from you! If you have any questions, suggestions, or feedback, please feel free to reach out to us at:
          <br />
          <a href="mailto:support@janasahayi.com">support@janasahayi.com</a>
        </p>
      </section>
    </div>
  );
};

export default AboutUs;
