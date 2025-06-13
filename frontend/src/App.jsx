import React, { useEffect, useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import './App.css';

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user"
};

const WS_URL = 'ws://127.0.0.1:8000/ws';

// Minimal preprocessing to match training data exactly
const preprocessImageSimple = (canvas, ctx, imageData) => {
  // The model was trained with NO preprocessing except resize + normalization
  // So we should do minimal processing here
  return imageData; // Return unprocessed data
};

const enhanceContrastMild = (imageData, width, height) => {
  // Disabled for now - model was trained without this
  return imageData;
};

// Image preprocessing utilities
const preprocessImage = (canvas, ctx, imageData) => {
  const width = canvas.width;
  const height = canvas.height;
  
  // Apply background removal using edge detection and skin tone filtering
  const processedData = removeBackground(imageData, width, height);
  
  // Apply noise reduction
  const denoisedData = applyGaussianBlur(processedData, width, height);
  
  // Enhance contrast for hand detection
  const enhancedData = enhanceContrast(denoisedData, width, height);
  
  return enhancedData;
};

const removeBackground = (imageData, width, height) => {
  const data = new Uint8ClampedArray(imageData.data);
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    
    // Skin tone detection (rough approximation)
    const isSkinTone = (
      r > 95 && g > 40 && b > 20 &&
      Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
      Math.abs(r - g) > 15 && r > g && r > b
    );
    
    // If not skin tone, make it black (background)
    if (!isSkinTone) {
      data[i] = 0;     // R
      data[i + 1] = 0; // G
      data[i + 2] = 0; // B
    }
  }
  
  return new ImageData(data, width, height);
};

const applyGaussianBlur = (imageData, width, height) => {
  const data = new Uint8ClampedArray(imageData.data);
  const result = new Uint8ClampedArray(data.length);
  
  // Simple 3x3 Gaussian kernel
  const kernel = [
    1/16, 2/16, 1/16,
    2/16, 4/16, 2/16,
    1/16, 2/16, 1/16
  ];
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      for (let channel = 0; channel < 3; channel++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = ((y + ky) * width + (x + kx)) * 4 + channel;
            sum += data[idx] * kernel[(ky + 1) * 3 + (kx + 1)];
          }
        }
        result[(y * width + x) * 4 + channel] = Math.min(255, Math.max(0, sum));
      }
      result[(y * width + x) * 4 + 3] = data[(y * width + x) * 4 + 3]; // Alpha
    }
  }
  
  return new ImageData(result, width, height);
};

const enhanceContrast = (imageData, width, height) => {
  const data = new Uint8ClampedArray(imageData.data);
  const factor = 1.5; // Contrast enhancement factor
  
  for (let i = 0; i < data.length; i += 4) {
    // Skip alpha channel
    for (let channel = 0; channel < 3; channel++) {
      let value = data[i + channel];
      value = ((value / 255 - 0.5) * factor + 0.5) * 255;
      data[i + channel] = Math.min(255, Math.max(0, value));
    }
  }
  
  return new ImageData(data, width, height);
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [currentPrediction, setCurrentPrediction] = useState("Waiting...");
  const [previousPrediction, setPreviousPrediction] = useState("None");
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [showProcessed, setShowProcessed] = useState(false);
  const [enableProcessing, setEnableProcessing] = useState(false);
  const socketRef = useRef(null);
  const intervalRef = useRef(null);

  // WebSocket connect
  const connectWebSocket = useCallback(() => {
    try {
      socketRef.current = new WebSocket(WS_URL);

      socketRef.current.onopen = () => {
        console.log("WebSocket connected");
        setIsConnected(true);
        setError(null);
      };

      socketRef.current.onmessage = (event) => {
        const newPrediction = event.data;
        console.log("Received:", newPrediction);

        setCurrentPrediction(prevCurrent => {
          setPreviousPrediction(prevCurrent);
          return newPrediction;
        });
      };

      socketRef.current.onclose = () => {
        console.log("WebSocket disconnected");
        setIsConnected(false);
      };

      socketRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setError("WebSocket connection failed");
        setIsConnected(false);
      };

    } catch (err) {
      console.error("Failed to create WebSocket:", err);
      setError("Failed to connect to server");
    }
  }, []);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [connectWebSocket]);

  // Send frames every 500ms
  useEffect(() => {
    const sendFrame = () => {
      if (
        webcamRef.current &&
        socketRef.current &&
        socketRef.current.readyState === WebSocket.OPEN
      ) {
        try {
          // Get video element from webcam
          const video = webcamRef.current.video;
          if (!video) {
            console.warn("Video element not available");
            return;
          }

          // Create canvas for image processing
          const canvas = canvasRef.current || document.createElement('canvas');
          if (!canvasRef.current) {
            canvasRef.current = canvas;
          }
          
          // Set canvas size to match video
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          
          const ctx = canvas.getContext('2d');
          
          // Draw video frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Get image data for processing
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          
          let finalImageData = imageData;
          
          if (enableProcessing) {
            // Apply preprocessing (simplified version)
            finalImageData = preprocessImageSimple(canvas, ctx, imageData);
          }
          
          // Put processed image back to canvas
          ctx.putImageData(finalImageData, 0, 0);
          
          // Crop to center square (hand region focus)
          const size = Math.min(canvas.width, canvas.height);
          const startX = (canvas.width - size) / 2;
          const startY = (canvas.height - size) / 2;
          
          // Create a new canvas for the cropped image
          const croppedCanvas = document.createElement('canvas');
          croppedCanvas.width = size;
          croppedCanvas.height = size;
          const croppedCtx = croppedCanvas.getContext('2d');
          
          croppedCtx.drawImage(canvas, startX, startY, size, size, 0, 0, size, size);
          
          // Convert to base64 and send
          const dataURL = croppedCanvas.toDataURL('image/jpeg', 0.8);
          
          if (dataURL) {
            console.log("Sending processed frame to backend");
            socketRef.current.send(dataURL);
            
            // Update preview canvas if showing processed view
            if (showProcessed && canvasRef.current) {
              const previewCtx = canvasRef.current.getContext('2d');
              previewCtx.drawImage(croppedCanvas, 0, 0, canvasRef.current.width, canvasRef.current.height);
            }
          } else {
            console.warn("No processed image generated");
          }
        } catch (err) {
          console.error("Error capturing/processing frame:", err);
        }
      }
    };

    if (isConnected) {
      intervalRef.current = setInterval(sendFrame, 500);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isConnected, showProcessed, enableProcessing]);

  const resetPredictions = () => {
    setPreviousPrediction("None");
    setCurrentPrediction("Waiting...");
  };

  const reconnect = () => {
    if (socketRef.current) {
      socketRef.current.close();
    }
    connectWebSocket();
  };

  return (
    <div className="container">
      <h1>ASL Translator (Enhanced with Background Removal)</h1>
      
      {/* Connection Status */}
      <div className={isConnected ? 'status-connected' : 'status-disconnected'} style={{ 
        padding: '15px', 
        borderRadius: '8px', 
        marginBottom: '20px',
        fontSize: '16px',
        fontWeight: 'bold'
      }}>
        🔗 Status: {isConnected ? 'Connected' : 'Disconnected'}
        {error && <div style={{ marginTop: '5px', fontSize: '14px' }}>❌ Error: {error}</div>}
      </div>

      <div className="video-container">
        {/* Original Webcam Feed */}
        <div className="video-item">
          <h3>📹 Original Feed</h3>
          <Webcam
            audio={false}
            height={240}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={320}
            videoConstraints={videoConstraints}
            mirrored={true}
            onUserMediaError={(error) => {
              console.error("Webcam error:", error);
              setError("Camera access denied or not available");
            }}
          />
        </div>

        {/* Processed Image Preview */}
        <div className="video-item">
          <h3>🖼️ Processed Feed (Sent to Model)</h3>
          <canvas
            ref={canvasRef}
            width={320}
            height={240}
          />
        </div>
      </div>
      
      <div className="checkbox-container">
        <label>
          <input
            type="checkbox"
            checked={showProcessed}
            onChange={(e) => setShowProcessed(e.target.checked)}
          />
          Show processed image preview
        </label>
      </div>

      <div className="checkbox-container">
        <label>
          <input
            type="checkbox"
            checked={enableProcessing}
            onChange={(e) => setEnableProcessing(e.target.checked)}
          />
          Enable image preprocessing (try disabling if getting wrong results)
        </label>
      </div>
      
      <div className="prediction-current">
        Current: {currentPrediction}
      </div>
      <div className="prediction-previous">
        Previous: {previousPrediction}
      </div>

      <div style={{ marginTop: '30px' }}>
        <button onClick={resetPredictions} style={{ marginRight: '10px' }}>
          🔄 Reset Predictions
        </button>
        <button onClick={reconnect} disabled={isConnected}>
          🔌 Reconnect WebSocket
        </button>
      </div>

      <div className="info-panel">
        <h4>🤖 Hand Detection Mode - Enhanced Accuracy</h4>
        <ul>
          <li>✨ <strong>MediaPipe Hand Detection:</strong> Backend automatically detects hands</li>
          <li>🎯 <strong>White Background:</strong> Creates clean, consistent background</li>
          <li>📸 <strong>Improved Processing:</strong> Matches training data format better</li>
          <li>🔄 <strong>Real-time:</strong> Processing happens automatically on each frame</li>
          <li>� <strong>Better Results:</strong> Should reduce "Delete" predictions significantly</li>
        </ul>
        <div className="tip">
          <strong>� New Feature:</strong> The backend now uses MediaPipe to detect your hand and create a clean white background. 
          This should dramatically improve prediction accuracy by providing consistent input to the model!
        </div>
      </div>
    </div>
  );
}

export default App;
