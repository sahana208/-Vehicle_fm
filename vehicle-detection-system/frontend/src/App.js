import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setProcessedImage(null);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data.results);
      setProcessedImage(response.data.image);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Vehicle Detection & License Plate Recognition</h1>
        <p>Upload an image to detect vehicles and extract license plate numbers</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <input
            type="file"
            id="image-upload"
            accept="image/*"
            onChange={handleFileSelect}
            className="file-input"
          />
          <label htmlFor="image-upload" className="file-label">
            {selectedFile ? selectedFile.name : 'Choose an image'}
          </label>
          
          {preview && (
            <div className="preview-container">
              <h3>Original Image</h3>
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}
          
          <button
            onClick={handleDetect}
            disabled={!selectedFile || loading}
            className="detect-button"
          >
            {loading ? 'Processing...' : 'Detect Vehicles'}
          </button>
          
          {error && <p className="error-message">{error}</p>}
        </div>

        {processedImage && (
          <div className="results-section">
            <h3>Processed Image</h3>
            <img src={processedImage} alt="Processed" className="processed-image" />
            
            {results && results.length > 0 && (
              <div className="detections-list">
                <h3>Detection Results</h3>
                {results.map((result, index) => (
                  <div key={index} className="detection-card">
                    <h4>Vehicle #{index + 1}</h4>
                    <p><strong>Type:</strong> {result.vehicle_type}</p>
                    <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
                    <p><strong>License Plate:</strong> {result.plate_number || 'Not detected'}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
