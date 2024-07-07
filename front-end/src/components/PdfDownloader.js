import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PDFList = () => {
  const [pdfFiles, setPdfFiles] = useState([]);

  useEffect(() => {
    fetchPDFs();
  }, []);

  const fetchPDFs = async () => {
    try {
      const response = await axios.get('http://localhost:5000/get_pdfs');
      setPdfFiles(response.data);
    } catch (error) {
      console.error('Error fetching PDFs:', error);
    }
  };

  const handleDownload = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/get_blob`, {
        responseType: 'blob'  //Ensure responseType is set to 'blob' to handle binary data
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));

      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) { 
      console.error(`Error downloading PDF ${filename}:`, error);
      //Optionally, provide user feedback about the error
    }
  };

  return (
    <div className="pdf-list-container">
      <h2>PDF Files</h2>
      {}
    </div>
  );
};

export default PDFList;
