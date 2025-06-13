# Sign Language Translator 🤟

> **🚧 Work in Progress** - Currently supports alphabet translation, with sentence translation coming soon!

A real-time American Sign Language (ASL) translator using computer vision and deep learning. This project captures hand gestures through a webcam and translates them into text using a custom-trained Convolutional Neural Network.

## 📋 Current Status & Roadmap

### ✅ Phase 1: Alphabet Translation (Current)
- **Status**: ✅ **Completed**
- **Capability**: Individual letter recognition (A-Z)
- **Accuracy**: 100% test accuracy achieved
- **Features**: Real-time letter detection with hand segmentation

### 🔄 Phase 2: Sentence Translation (Coming Soon)
- **Status**: 🚧 **In Development**
- **Planned Features**:
  - Word formation from letter sequences
  - Common word and phrase recognition
  - Sentence structure understanding
  - Grammar and context processing
  - Multi-word gesture recognition

### 🔮 Future Phases
- **Phase 3**: Advanced conversational ASL
- **Phase 4**: Bidirectional translation (text-to-sign)
- **Phase 5**: Mobile app development

## 🎯 Features

- **Real-time Letter Recognition**: Live webcam feed processing for instant alphabet gesture recognition
- **Individual Letter Detection**: Currently supports A-Z alphabet translation (Phase 1)
- **Hand Detection**: Advanced MediaPipe integration for accurate hand segmentation
- **Clean Background Processing**: Automatic white background generation for improved accuracy
- **29 Sign Classes**: Supports A-Z letters plus Space, Delete, and Nothing gestures
- **High Accuracy**: Achieved **100% test accuracy** on alphabet validation data
- **Modern UI**: Clean, responsive React-based frontend
- **WebSocket Communication**: Real-time bidirectional communication between frontend and backend
- **🔄 Future**: Sentence translation capabilities in development

## 🏆 Model Performance

Our TinyVGG model achieved exceptional performance during training:

### Training Metrics (5 Epochs)
| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-------|------------|----------------|-----------|---------------|
| 1     | 1.253      | 61.79%         | 0.099     | 96.43%        |
| 2     | 0.290      | 90.21%         | 0.080     | 96.43%        |
| 3     | 0.192      | 93.44%         | 0.182     | 96.43%        |
| 4     | 0.147      | 94.94%         | 0.002     | **100.00%**   |
| 5     | 0.123      | 95.77%         | 0.0004    | **100.00%**   |

### Final Results
- **Test Accuracy**: 100% 🎉
- **Training Accuracy**: 95.77%
- **Model Size**: Lightweight TinyVGG architecture
- **Inference Speed**: Real-time processing capability

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for the API
- **PyTorch**: Deep learning framework for model training and inference
- **MediaPipe**: Hand detection and landmark extraction
- **OpenCV**: Computer vision preprocessing
- **WebSockets**: Real-time communication

### Frontend
- **React**: Modern JavaScript framework
- **Vite**: Fast build tool and development server
- **WebRTC**: Browser-based webcam access
- **Canvas API**: Real-time image processing

### Model Architecture
- **TinyVGG**: Custom CNN architecture
- **Input**: 64x64 RGB images
- **Layers**: 3 convolutional blocks + fully connected layer
- **Output**: 29 classes (A-Z + Space, Delete, Nothing)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam-enabled device

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sign_lang_translator/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requiremens.txt
   ```

4. **Run the backend server**
   ```bash
   python main.py
   ```
   Server will start at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   Frontend will be available at `http://localhost:5173`

## 📁 Project Structure

```
sign_lang_translator/
├── backend/
│   ├── main.py                 # FastAPI server with WebSocket endpoints
│   ├── model_class/
│   │   └── model_0.py         # TinyVGG model architecture
│   ├── models/
│   │   └── model_0.pth        # Trained model weights
│   ├── dataset/               # Training dataset
│   ├── v_1.ipynb             # Model training notebook
│   └── requiremens.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── App.css           # Styling
│   │   └── main.jsx          # Entry point
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Vite configuration
└── README.md                 # This file
```

## 🔧 How It Works (Current Implementation)

**Phase 1 - Alphabet Recognition Pipeline:**

1. **Webcam Capture**: Frontend captures live video from user's webcam
2. **Frame Processing**: Each frame is sent to the backend via WebSocket
3. **Hand Detection**: MediaPipe detects hand landmarks and creates a mask
4. **Background Removal**: Hand is isolated on a clean white background
5. **Preprocessing**: Image is resized to 64x64 and normalized
6. **Model Inference**: TinyVGG model predicts the letter class
7. **Real-time Display**: Letter prediction is sent back and displayed instantly

**Phase 2 - Planned Sentence Translation Pipeline:**
- Letter sequence collection and word formation
- Context analysis and grammar processing
- Sentence structure recognition
- Enhanced multi-gesture understanding

## 🎯 Supported Signs (Current Phase)

**Phase 1 - Alphabet Translation:**
The model currently recognizes 29 different classes for individual letter recognition:
- **Letters**: A through Z (26 classes)
- **Actions**: Space, Delete
- **Neutral**: Nothing (no sign detected)

**Coming in Phase 2 - Sentence Translation:**
- Word formation from letter sequences
- Common ASL words and phrases
- Contextual sentence understanding
- Grammar processing

## 🔍 Key Features

### Advanced Hand Detection
- **MediaPipe Integration**: Robust hand landmark detection
- **Automatic Segmentation**: Creates clean hand masks
- **White Background**: Consistent background for improved accuracy
- **Fallback Processing**: Skin tone detection when MediaPipe fails

### Real-time Processing
- **WebSocket Communication**: Low-latency bidirectional communication
- **Optimized Pipeline**: Efficient preprocessing and inference
- **Error Handling**: Graceful fallbacks and error recovery

### Model Architecture
```python
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (conv_block_2): Sequential(...)
  (conv_block_3): Sequential(...)
  (fc_layer): Sequential(
    (0): Flatten()
    (1): Linear(in_features=640, out_features=29)
  )
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Delete" predictions**: Ensure good lighting and clear hand visibility
2. **WebSocket connection issues**: Check if backend server is running on port 8000
3. **Webcam access denied**: Grant camera permissions in browser settings
4. **Model loading errors**: Verify model file exists at `backend/models/model_0.pth`

### Performance Tips

- Use good lighting conditions
- Keep hand clearly visible in frame
- Avoid cluttered backgrounds (the system creates white background automatically)
- Make distinct, clear gestures

## 📊 Training Details

### Dataset
- **Size**: Multiple thousand images per class
- **Classes**: 29 total (A-Z + Space, Delete, Nothing)
- **Preprocessing**: Resize to 64x64, normalize to [0,1]
- **Data Augmentation**: None (model achieved 100% without augmentation)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: Variable
- **Epochs**: 5
- **Device**: CUDA-enabled GPU (if available)

## 🤝 Contributing

We welcome contributions to help advance this project from alphabet to sentence translation!

### Current Priorities
- **Phase 2 Development**: Help implement sentence translation features
- **Dataset Expansion**: Contribute word and phrase datasets
- **Model Improvements**: Enhance accuracy for complex gestures
- **UI/UX**: Improve user interface for sentence display

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Areas
- **Machine Learning**: Model architecture improvements for sentence understanding
- **Data Science**: Dataset collection and preprocessing for words/phrases
- **Frontend**: Enhanced UI for sentence display and interaction
- **Backend**: Improved processing pipeline for complex translations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: For excellent hand detection capabilities
- **PyTorch**: For the deep learning framework
- **FastAPI**: For the modern web framework
- **React**: For the frontend framework

## 📞 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the development team.

---

**🚧 Currently in Phase 1: Alphabet Translation Complete**  
**🔄 Phase 2: Sentence Translation Coming Soon**  
**Built with ❤️ for the deaf and hard-of-hearing community**
