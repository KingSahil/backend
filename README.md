# YouTube Transcript & Chapter Generator API 🎥

A high-performance Python backend API powered by FastAPI that fetches YouTube video transcripts and leverages AI to generate intelligent chapters, summaries, quizzes, and answer student questions.

## ✨ Features

- 📝 **YouTube Transcript Extraction**: Fetch transcripts in multiple languages
- 🤖 **AI-Powered Chapter Generation**: Automatically create timestamped chapters with summaries
- 📊 **Smart Video Summarization**: Generate comprehensive video summaries
- 💬 **AI Question Answering**: Answer student questions based on video content
- 📝 **Quiz Generation**: Create multiple-choice quizzes from video transcripts
- 🔄 **Automatic API Fallback**: Gemini → OpenRouter failover for high availability
- 🚀 **Fast & Async**: Built with FastAPI for optimal performance
- 🌐 **CORS Enabled**: Ready for cross-origin requests
- 🔌 **Dual AI Provider Support**: 
  - Google Gemini (gemini-2.0-flash-exp)
  - OpenRouter (Claude 3 Haiku and other models)

## 🛠️ Tech Stack

- **Framework**: FastAPI 0.109.0
- **AI APIs**: 
  - Google Generative AI (Gemini)
  - OpenRouter (Claude, GPT, etc.)
- **YouTube API**: youtube-transcript-api 0.6.2
- **HTTP Client**: httpx 0.26.0
- **Server**: uvicorn 0.27.0
- **Validation**: pydantic 2.5.3

## 📋 Prerequisites

- Python 3.8 or higher
- **Gemini API Key** ([Get free quota here](https://ai.google.dev/))
- **OpenRouter API Key** (Optional, for fallback) ([Get one here](https://openrouter.ai/))

## 🚀 Installation

1. **Navigate to the backend directory**:
```bash
cd backend
```

2. **Create a virtual environment**:
```bash
python -m venv venv
```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Set up environment variables**:
   
   Create a `.env` file in the `backend` directory:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   # Required: Gemini API Key (Primary AI provider)
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional: OpenRouter API Key (Fallback when Gemini quota is exceeded)
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # Server port (default: 8000)
   PORT=8000
   ```

## 🎯 Usage

### Start the Server

```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 📚 Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs (Interactive API testing)
- **ReDoc**: http://localhost:8000/redoc (Clean documentation)

## 🔌 API Endpoints

### 1. `POST /analyze` - Analyze Video & Generate Chapters

Fetch transcript and generate AI-powered chapters with timestamps.

**Request:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "model": "gemini-2.0-flash-exp",
  "languages": ["en"],
  "api_provider": "gemini"
}
```

**Parameters:**
- `video_url` (required): YouTube video URL
- `model` (optional): AI model to use (default: varies by provider)
- `languages` (optional): Preferred transcript languages (default: auto-detect)
- `api_provider` (optional): "gemini" or "openrouter" (default: "gemini")

**Response:**
```json
{
  "video_id": "VIDEO_ID",
  "transcript": [
    {
      "text": "Welcome to this tutorial...",
      "start": 0.0,
      "duration": 2.5
    }
  ],
  "chapters": [
    {
      "timestamp": "00:00",
      "title": "Introduction",
      "summary": "Overview of the tutorial and what will be covered"
    },
    {
      "timestamp": "02:30",
      "title": "Setup Instructions",
      "summary": "Step-by-step guide to setting up the development environment"
    }
  ],
  "summary": "A comprehensive tutorial covering setup, implementation, and best practices"
}
```

---

### 2. `POST /ai-question` - Ask Questions About Video

Get AI-powered answers based on video transcript content.

**Request:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_title": "Python Tutorial for Beginners",
  "question": "What is a variable in Python?",
  "api_provider": "gemini"
}
```

**Response:**
```json
{
  "answer": "Based on the video, a variable in Python is a container that stores data values. The instructor explained that variables are created using the assignment operator (=) and don't require type declaration. For example:\n\n- `name = \"John\"` creates a string variable\n- `age = 25` creates an integer variable\n\nThe video also mentioned that Python is dynamically typed, meaning variable types are determined automatically at runtime."
}
```

---

### 3. `POST /generate-quiz` - Generate Video Quiz

Create a multiple-choice quiz based on video content.

**Request:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_title": "Introduction to Machine Learning",
  "api_provider": "gemini"
}
```

**Response:**
```json
{
  "quiz": [
    {
      "question": "What is supervised learning according to the video?",
      "options": [
        "A) Learning without labeled data",
        "B) Learning from labeled training data to make predictions",
        "C) Learning by trial and error",
        "D) Learning by clustering similar data"
      ],
      "correct": "B"
    }
  ]
}
```

---

### 4. `GET /transcript/{video_id}` - Get Transcript Only

Fetch only the transcript without AI processing.

**Response:**
```json
{
  "video_id": "VIDEO_ID",
  "transcript": [
    {
      "text": "Transcript text",
      "start": 0.0,
      "duration": 2.5
    }
  ]
}
```

---

### 5. `GET /health` - Health Check

Check API status.

**Response:**
```json
{
  "status": "healthy"
}
```

## 🤖 AI Provider Configuration

### Gemini (Default, Recommended)
- **Model**: `gemini-2.0-flash-exp`
- **Quota**: Free tier available
- **Speed**: Very fast
- **Cost**: Free quota generous for development

### OpenRouter (Fallback)
- **Default Model**: `anthropic/claude-3-haiku`
- **Alternative Models**: 
  - `openai/gpt-4-turbo`
  - `openai/gpt-3.5-turbo`
  - `google/gemini-pro`
  - [View all models](https://openrouter.ai/models)

### Automatic Fallback System

The API automatically handles quota exhaustion:
1. Tries **Gemini** first (fast & free)
2. Falls back to **OpenRouter** if Gemini quota exceeded
3. Returns error only if both providers fail

**To enable fallback**, ensure both API keys are configured in `.env`

## 📖 Example Usage

### Using cURL

```bash
# Analyze video with chapter generation
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "api_provider": "gemini"
  }'

# Ask a question about the video
curl -X POST "http://localhost:8000/ai-question" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "video_title": "Sample Video",
    "question": "What is the main topic?"
  }'

# Generate a quiz
curl -X POST "http://localhost:8000/generate-quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "video_title": "Sample Video"
  }'
```

### Using Python

```python
import requests

# Analyze video
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "api_provider": "gemini"
    }
)

data = response.json()
print(f"Summary: {data['summary']}\n")
print("Chapters:")
for chapter in data['chapters']:
    print(f"  [{chapter['timestamp']}] {chapter['title']}")
    print(f"    {chapter['summary']}\n")
```

### Using JavaScript/TypeScript

```javascript
// Analyze video
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    video_url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    api_provider: 'gemini'
  })
});

const data = await response.json();
console.log('Chapters:', data.chapters);
```

### Using React Native (Expo)

```javascript
import { learningHubService } from './lib/learningHub';

// From your React Native app
const chapters = await learningHubService.generateChapters(videoUrl);
const answer = await learningHubService.askVideoQuestion(videoUrl, title, question);
const quiz = await learningHubService.generateQuiz(videoUrl, title);
```

## 📁 Project Structure

```
backend/
├── main.py              # FastAPI application with all endpoints
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .env.example        # Environment template
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## ⚠️ Error Handling

The API handles various error scenarios gracefully:

| Error | Status Code | Description |
|-------|-------------|-------------|
| Invalid YouTube URL | 400 | URL format is incorrect |
| No Transcript Found | 404 | Video has no captions/transcript |
| Transcripts Disabled | 404 | Creator disabled transcripts |
| Gemini Quota Exceeded | 503* | Falls back to OpenRouter if configured |
| OpenRouter API Error | 500 | OpenRouter request failed |
| AI Response Parse Error | 500 | Failed to parse AI-generated JSON |

*If both providers fail, returns 503 with instructions

## 🔒 Security Best Practices

1. **Never commit `.env` file** - Keep API keys secret
2. **Use environment variables** - Don't hardcode keys
3. **Rotate API keys regularly** - Update keys periodically
4. **Monitor API usage** - Track quota and costs
5. **Rate limiting** - Consider implementing rate limits for production

## 🐛 Troubleshooting

### Import Error: youtube_transcript_api

**Error**: `ModuleNotFoundError: No module named 'youtube_transcript_api'`

**Solution**:
```bash
pip install youtube-transcript-api==0.6.2
```

### Gemini Quota Exceeded

**Error**: `503 - Gemini quota exceeded`

**Solution**:
1. Wait for quota reset (daily limits)
2. Add OpenRouter API key to `.env` for automatic fallback
3. Or manually set `api_provider: "openrouter"` in requests

### CORS Issues

If accessing from a web browser, ensure CORS is properly configured. The API allows all origins by default (`allow_origins=["*"]`). For production, restrict to specific domains.

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Change port in .env
PORT=8001

# Or specify when running
uvicorn main:app --port 8001
```

## 🚀 Deployment

### Deploy to Railway

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables in Railway dashboard
4. Deploy automatically on push

### Deploy to Render

1. Create account at [render.com](https://render.com)
2. Create new Web Service from GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables

### Deploy to Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your_key
heroku config:set OPENROUTER_API_KEY=your_key
git push heroku main
```

## 📊 Performance

- **Average Response Time**: 2-5 seconds (depends on video length)
- **Transcript Fetch**: ~500ms
- **AI Processing**: 1.5-4 seconds
- **Max Transcript Length**: 10,000 characters (for AI processing)
- **Supported Video Lengths**: Up to 3+ hours

## 💻 Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🔗 Related Projects

- **Frontend**: React Native Expo app (in `/frontend`)
- **Database**: Supabase integration with AI cache
- **Authentication**: Auth0 for user management

## 💡 Use Cases

- **Educational Platforms**: Help students understand video lectures
- **Content Creation**: Generate chapter markers for YouTube videos
- **Research**: Extract key insights from educational content
- **Accessibility**: Make video content more navigable and searchable
- **Study Tools**: Create quizzes and Q&A from educational videos

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the [AI-GUIDES](../AI-GUIDES/) directory for detailed documentation
- Review [TROUBLESHOOTING.md](../AI-GUIDES/TROUBLESHOOTING.md)

---

**Built with ❤️ using FastAPI, Google Gemini, and OpenRouter**
