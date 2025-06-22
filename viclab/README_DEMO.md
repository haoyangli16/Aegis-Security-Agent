# VicLab Real-time Video Analysis Demo

A complete real-time video analysis system with SmolVLM integration, featuring a FastAPI backend and React frontend.

## 🚀 Quick Start

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd VicLab
   pip install -r requirements.txt
   ```

2. **Start the backend server:**
   ```bash
   # Option 1: Using the startup script
   python start_backend.py
   
   # Option 2: Using uvicorn directly
   uvicorn api.main:app --reload --port 8000
   ```

3. **Verify backend is running:**
   - Open http://localhost:8000 in your browser
   - Check API docs at http://localhost:8000/docs

### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the frontend:**
   ```bash
   npm start
   ```

3. **Open the demo:**
   - Frontend will be available at http://localhost:3000
   - Make sure your camera permissions are enabled

## 📋 Features

### Real-time Video Analysis
- **Live Camera Stream**: Real-time webcam feed display
- **Frame Buffer**: Collects video frames in a rolling buffer
- **Single Frame Analysis**: Analyze the current frame with custom prompts
- **Multi-frame Analysis**: Analyze recent frames from the buffer
- **Stream Status**: Real-time buffer status monitoring

### API Endpoints

#### Core Analysis
- `POST /api/analyze-frame` - Analyze a single image/frame
- `POST /api/analyze-recent-frames` - Analyze recent frames from buffer

#### Stream Management
- `POST /api/start-stream` - Start video stream buffer
- `POST /api/stop-stream` - Stop video stream buffer
- `GET /api/stream-status` - Get buffer status

#### Video Upload
- `POST /api/upload-video` - Upload and optionally analyze video files
- `GET /api/videos` - List uploaded videos
- `DELETE /api/videos/{filename}` - Delete uploaded videos

#### Health Check
- `GET /` - API status
- `GET /health` - Health check

## 🎮 How to Use

1. **Start the Backend:**
   ```bash
   python start_backend.py
   ```

2. **Start the Frontend:**
   ```bash
   cd frontend
   npm start
   ```

3. **Use the Demo:**
   - Open http://localhost:3000
   - Allow camera access when prompted
   - Click "Start Stream Buffer" to begin collecting frames
   - Enter a prompt (e.g., "Describe what you see")
   - Use "Analyze Current Frame" for immediate analysis
   - Use "Analyze Recent Frames" for buffer analysis

## 🧪 Testing Examples

### Test Prompts
- "Describe what you see in the video"
- "Count the number of people in the frame"
- "What objects are visible?"
- "Describe the scene and any movement"
- "What colors are dominant in the image?"

### API Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Start stream buffer
curl -X POST http://localhost:8000/api/start-stream \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "video_source=0&fps_limit=10"

# Check stream status
curl http://localhost:8000/api/stream-status

# Stop stream
curl -X POST http://localhost:8000/api/stop-stream
```

## 🔧 Troubleshooting

### Backend Issues

1. **Import Errors:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Issues:**
   - Ensure you have sufficient GPU memory (4GB+ recommended)
   - Models will be downloaded on first run

3. **Port Already in Use:**
   ```bash
   # Kill process on port 8000
   kill -9 $(lsof -t -i:8000)
   ```

### Frontend Issues

1. **Camera Access Denied:**
   - Check browser permissions
   - Use HTTPS in production

2. **API Connection Failed:**
   - Verify backend is running on port 8000
   - Check CORS settings

3. **TypeScript Errors:**
   ```bash
   cd frontend
   npm install --force
   ```

## 📁 Project Structure

```
VicLab/
├── api/                          # FastAPI backend
│   ├── main.py                   # Main API app
│   ├── analyze_stream.py         # Video analysis endpoints
│   └── upload_video.py           # Video upload endpoints
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── App.tsx              # Main React component
│   │   └── ...
│   └── package.json
├── viclab/                      # Core VicLab library
│   └── video/
│       └── realtime_video.py    # SmolVLM processor
├── requirements.txt             # Python dependencies
├── start_backend.py            # Backend startup script
└── video_analysis_quick_start.py # CLI examples
```

## 🚀 Development

### Adding New Features

1. **New API Endpoints:** Add to `api/analyze_stream.py` or `api/upload_video.py`
2. **Frontend Components:** Modify `frontend/src/App.tsx`
3. **Video Processors:** Extend `viclab/video/realtime_video.py`

### Performance Tips

- Adjust `fps_limit` based on your hardware capabilities
- Reduce `max_frames_buffer` if memory is limited
- Use GPU acceleration when available

## 📝 Notes

- The system uses SmolVLM2-256M-Video-Instruct model
- First run will download model weights (~500MB)
- Requires camera permissions for live analysis
- Buffer analysis requires at least 3 frames

## 🐛 Known Issues

- TypeScript configuration may need adjustment for React 19
- Some browsers may require HTTPS for camera access
- Model initialization can take 30-60 seconds on first run

## 📞 Support

If you encounter issues:

1. Check the console logs in both backend and frontend
2. Verify all dependencies are installed
3. Ensure camera permissions are granted
4. Check that ports 3000 and 8000 are available 