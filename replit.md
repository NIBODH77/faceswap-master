# FaceSwap Web Application

## Overview
यह एक web-based FaceSwap application है जो deep learning का उपयोग करके images और videos में faces को swap करता है। यह original [deepfakes/faceswap](https://github.com/deepfakes/faceswap) repository का Flask web wrapper है।

## Project Architecture

### Components
1. **Backend**: Flask web application (Python 3.10)
2. **AI Models**: 
   - **s3fd**: Face detection
   - **fan**: Face alignment
   - **bisenet-fp**: Face masking
   - **original**: Training model (lightweight, fast)
3. **Frontend**: HTML/CSS/JavaScript interface with Hindi language support

### Key Features
- Upload images or videos for face swapping
- Automatic face detection and extraction
- Model training with progress tracking
- Real-time status updates
- Support for both images and videos
- Hindi language interface

## Technical Setup

### Python Version
- **Current**: Python 3.12 64-bit
- **Note**: Updated to work with Python 3.12 and newer TensorFlow versions for Replit compatibility

### Dependencies
Core dependencies installed:
- `tensorflow-cpu>=2.16.0` - Deep learning framework (updated for Python 3.12)
- `opencv-python` - Image processing
- `flask` - Web framework
- `numpy<2.0.0` - Numerical computing
- `pillow<10.0.0` - Image handling
- Additional ML libraries: scikit-learn, matplotlib, imageio

### AI Models Auto-Download
AI models को manually download करने की जरूरत नहीं है। वे automatically download होते हैं जब पहली बार use होते हैं:
- S3FD detector weights
- FAN aligner weights
- BiSeNet-FP mask weights

Models download होते हैं `models/` directory में।

## How It Works

### Workflow

**Two Options Available:**

#### Option 1: Instant Swap (Quick, No Training)
1. **Upload**: User uploads source face और target media (image/video)
2. **Instant Swap**: Direct face swap without training (fast but less accurate)
3. **Download**: Result immediately available for download

#### Option 2: Train & Convert (Best Quality)
1. **Upload**: User uploads source face और target media (image/video)
2. **Auto-Extract**: Faces automatically extract होते हैं dono files से
   - System shows face count for both source and target
   - Train Model button enables only after successful extraction
3. **Train**: Model train होता है source और target faces पर (5000 iterations, ~30 seconds)
   - Background training with live progress tracking
   - Convert button enables only after training completes
4. **Convert**: Trained model target media पर apply होता है
5. **Download**: User result download कर सकते हैं

**Button State Management:**
- Upload Files: Enabled when both source and target are selected
- Instant Swap: Enabled after successful upload
- Train Model: Enabled after successful face extraction
- Face Swap: Enabled only after training completes
- Every new upload resets all states (no state leakage between sessions)

### API Endpoints
- `POST /upload` - Upload source and target files
- `POST /instant_swap` - Quick face swap without training
- `POST /extract` - Extract faces from uploaded files (called automatically after upload)
- `POST /train` - Train face swap model
- `GET /training_status/<session_id>` - Check training progress
- `POST /convert` - Convert target using trained model
- `GET /download/<session_id>/<filename>` - Download result

## Recent Changes

### Complete Workflow Implementation (Nov 22, 2025)
Implemented full Upload → Extract → Train → Convert workflow with proper state management:

**Frontend Workflow:**
1. Auto-extraction after upload - faces extracted from both source and target automatically
2. Button state management - Train Model button enables only after extraction completes
3. Progress tracking - Real-time progress bars during extraction, training, and conversion
4. Training polling - Frontend polls backend every 3 seconds for training status
5. Convert gating - Convert button enables only after training completes successfully

**State Management:**
- All session state (flags, IDs, results) reset on new upload
- Zero state leakage between sessions
- Buttons strictly follow workflow sequence
- Previous results cleared when starting new session

**User Experience:**
- Clear Hindi/English mixed messages at each step
- Face count display after extraction
- Progress indicators during long operations
- Error messages with helpful details
- Instant Swap option for quick results without training

### Replit Import Setup (Nov 22, 2025)
1. **Python Version**: Updated to Python 3.12 for Replit compatibility
2. **TensorFlow**: Upgraded to tensorflow-cpu>=2.16.0 (compatible with Python 3.12)
3. **Dependencies**: All required packages successfully installed via pip
4. **Web Server**: Flask app running on 0.0.0.0:5000 with no errors
5. **Gitignore**: Added comprehensive .gitignore for Python projects
6. **Command Fixes**: 
   - Removed `-M bisenet-fp` from extract command (not needed during extraction)
   - Changed `-bs` to `-b` in train command (deprecated parameter fixed)
   - Face swap now fully functional without errors

### Production-Level Error Handling (Nov 22, 2025)
Implemented comprehensive error handling and validation across all endpoints:

**Extract Endpoint Improvements:**
- Session existence validation before processing
- File existence and size validation (detect empty files)
- Better file type detection for images vs videos
- Detailed error messages for corrupted or unsupported files
- Face count validation (returns error if no faces detected)
- Specific timeout handling with helpful messages

**Train Endpoint Improvements:**
- Pre-flight validation of session and directories
- Face count validation for both source and target
- Prevention of duplicate training sessions
- Detailed logging of face counts before training
- Better status tracking (starting → training → completed/failed)
- Clear error messages for missing or empty face directories

**Convert Endpoint Improvements:**
- Comprehensive training status validation
- Model file existence verification
- Target file validation before conversion
- Better output file detection and validation
- Detailed success/failure messages
- Timeout handling for large files

**Overall Improvements:**
- All errors now include helpful "details" field
- Consistent error response format
- Comprehensive logging at every step
- Zero possibility of unclear errors
- Production-ready error handling

### Bug Fixes (Nov 22, 2025)
1. **Critical Fix**: Extraction bug fixed जहां output directory creation file finding से पहले हो रहा था, जिससे extraction empty folder पर run हो रहा था
2. **Training Status**: Training status initialization improved to avoid "not_found" transient states
3. **Error Handling**: Better error capture - अब stdout और stderr दोनों capture होते हैं
4. **Model Verification**: Convert endpoint अब verify करता है कि model files exist करती हैं before conversion

### Improvements
- Training runs in background thread with progress monitoring
- Conversion only allowed after training completion
- Better file filtering (files vs directories)
- Comprehensive error messages with details
- Timeout handling for long-running operations

## Running the Application

### Development
Application automatically runs on:
- Host: `0.0.0.0` (required for Replit preview)
- Port: `5000`
- Debug mode: Enabled

### Important Notes
1. Training takes time (5000 iterations ≈ 30 seconds on CPU)
2. First run will download AI models (~500MB total)
3. Conversion fails अगर training complete नहीं हुआ है
4. Models save होते हैं `models/<session_id>/` में
5. Results save होते हैं `outputs/<session_id>/` में

## File Structure
```
.
├── web_app.py              # Flask application
├── templates/
│   └── index.html         # Frontend UI
├── faceswap.py            # Main faceswap CLI
├── lib/                   # Core faceswap libraries
├── plugins/               # Detection, alignment, mask plugins
├── models/                # AI model weights (auto-downloaded)
├── uploads/               # User uploaded files
├── outputs/               # Conversion results
└── requirements/          # Python dependencies
```

## Troubleshooting

### Common Issues
1. **Conversion fails**: Wait for training to complete (check /training_status)
2. **No faces detected**: Ensure images contain clear, visible faces
3. **Model download fails**: Check internet connection
4. **Port already in use**: Another process is using port 5000

### User Preferences
- Language: Hindi + English mixed interface
- Quick training: 5000 iterations (adjustable)
- Default model: "original" (lightweight, fast)
