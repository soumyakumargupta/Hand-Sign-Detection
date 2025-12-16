# Hand Sign Detection - Flask Application

A real-time hand sign detection web application built with Flask, MediaPipe, and TensorFlow. This application can recognize American Sign Language (ASL) alphabets through your webcam.

## Features

- Real-time hand detection using MediaPipe
- ASL alphabet recognition (A-Z)
- Web-based interface using Flask
- LSTM-based deep learning model for gesture classification
- Live accuracy display

## Requirements

- Python 3.8 - 3.12 (recommended: Python 3.12)
- Webcam/Camera
- Windows/Linux/macOS

## Installation

1. Clone this repository:
```bash
git clone https://github.com/soumyakumargupta/Hand-Sign-Detection.git
cd Hand-Sign-Detection
```

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv312
venv312\Scripts\activate

# Linux/macOS
python3 -m venv venv312
source venv312/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with MediaPipe or TensorFlow, ensure you're using Python 3.8-3.12. Python 3.13+ is not yet supported by MediaPipe.

## Usage

### Running the Flask Web Application

1. Activate the virtual environment (if not already activated):
```bash
# Windows
venv312\Scripts\activate

# Linux/macOS
source venv312/bin/activate
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and go to `http://localhost:5000`

4. Allow camera permissions when prompted

5. Show hand signs in front of the camera to see real-time predictions

### Running the Demo Script

Alternatively, you can run the demo script:
```bash
python demo.py
```

### Stopping the Application

- Press `Ctrl+C` in the terminal to stop the Flask server
- Deactivate the virtual environment: `deactivate`

## Project Structure

```
Hand-Sign-Detection/
├── app.py                 # Main Flask application
├── demo.py               # Alternative demo script
├── alphabets.h5          # Trained LSTM model weights
├── 0.npy                 # Training data (numpy array)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore file
├── templates/
│   └── index.html       # Web interface template
└── Alphabets/           # Training data directory
    ├── A/               # Hand sign data for letter A
    ├── B/               # Hand sign data for letter B
    └── ...              # Data for other letters
```

## How It Works

1. **Hand Detection**: Uses MediaPipe to detect hand landmarks in real-time
2. **Feature Extraction**: Extracts 126 keypoints (21 landmarks × 3 coordinates × 2 hands)
3. **Sequence Processing**: Collects 30 frames of hand data for temporal analysis
4. **Classification**: LSTM model predicts the most likely alphabet based on hand movements
5. **Confidence Filtering**: Only shows predictions above 70% confidence
6. **Stability Check**: Confirms prediction by requiring 5 seconds of consistent results

## Model Architecture

- **Input**: 30 frames × 126 keypoints
- **LSTM Layers**: 64 → 128 → 64 units with ReLU activation
- **Dense Layers**: 64 → 32 → 26 (alphabet classes) with softmax output
- **Optimizer**: Adam
- **Loss**: Categorical crossentropy

## Technical Details

- **Framework**: Flask for web server
- **Computer Vision**: OpenCV for video processing
- **Hand Detection**: MediaPipe Hands solution
- **Deep Learning**: TensorFlow/Keras LSTM model
- **Real-time Processing**: 30 FPS video stream

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Try changing the camera index in `app.py` (line 13): `cv2.VideoCapture(1)` for external cameras

### Model Loading Issues
- Ensure `alphabets.h5` is in the same directory as `app.py`
- Check that all dependencies are properly installed

### Performance Issues
- Reduce video quality for better performance on slower machines
- Ensure good lighting conditions for better hand detection

### Python Version Issues
- If MediaPipe installation fails, check your Python version
- Use Python 3.8 to 3.12 (Python 3.13+ not yet supported by MediaPipe)
- Create a fresh virtual environment with a compatible Python version

### Virtual Environment Not Activating
**Windows:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv312\Scripts\activate
```

**Linux/macOS:**
```bash
chmod +x venv312/bin/activate
source venv312/bin/activate
```

## Dependencies

All dependencies are listed in `requirements.txt`:
- Flask 3.0.0
- OpenCV 4.10.0.84
- NumPy 1.26.4
- MediaPipe 0.10.14
- TensorFlow 2.17.0

**Note:** The virtual environment folder (`venv312/`, `.venv/`, etc.) is excluded from Git via `.gitignore`. Each user should create their own virtual environment locally.
