from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import os
from dotenv import load_dotenv
import httpx
from typing import List, Optional
import re
import google.genai as genai
from google.genai import types
import json
# Load environment variables
load_dotenv()

app = FastAPI(
    title="YouTube Transcript & Chapter Generator API",
    description="API to fetch YouTube transcripts and generate chapters using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class VideoRequest(BaseModel):
    video_url: str
    model: Optional[str] = "anthropic/claude-3-haiku"
    languages: Optional[List[str]] = None  # e.g., ['hi', 'en', 'es']
    api_provider: Optional[str] = "openrouter"  # 'openrouter' or 'gemini'

class AIQuestionRequest(BaseModel):
    video_url: str
    video_title: str
    question: str
    api_provider: Optional[str] = "gemini"  # 'openrouter' or 'gemini'

class QuizRequest(BaseModel):
    video_url: str
    video_title: str
    api_provider: Optional[str] = "gemini"  # 'openrouter' or 'gemini'

class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float

class Chapter(BaseModel):
    timestamp: str
    title: str
    summary: str

class VideoResponse(BaseModel):
    video_id: str
    transcript: List[TranscriptSegment]
    chapters: List[Chapter]
    summary: str

# Helper functions
def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

async def call_openrouter(transcript_text: str, model: str, video_duration: str = None, video_duration_seconds: float = None) -> dict:
    """Call OpenRouter API to generate chapters and summary"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    duration_info = f"\nVideo Duration: {video_duration} (max {int(video_duration_seconds)} seconds)" if video_duration else ""
    
    prompt = f"""You are a helpful assistant that analyzes YouTube video transcripts and creates structured chapters with summaries.
{duration_info}

Given the following video transcript with timestamps in [MM:SS] or [HH:MM:SS] format, please:
1. Identify major topic changes and create 5-8 chapters
2. For each chapter, YOU MUST extract the EXACT timestamp_seconds from the [HH:MM:SS] or [MM:SS] markers in the transcript
3. Parse the timestamps like this: [00:45] = 45 seconds, [02:30] = 150 seconds, [01:15:20] = 4520 seconds
4. CRITICAL: All timestamp_seconds MUST be between 0 and {int(video_duration_seconds)} (the video duration)
5. Use timestamps that actually appear in the transcript - do NOT make up timestamps
6. Provide a descriptive title and brief summary (2-3 sentences) for each chapter
7. Create an overall video summary (3-4 sentences)

Transcript:
{transcript_text}

Please respond in the following JSON format:
{{
    "chapters": [
        {{
            "timestamp_seconds": 0,
            "title": "Chapter Title",
            "summary": "Brief summary of this chapter"
        }}
    ],
    "overall_summary": "Overall video summary"
}}"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print(f"🔄 Calling OpenRouter API with model: {model}")
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Check response status
            if not response.is_success:
                error_text = response.text
                print(f"❌ OpenRouter API error: {response.status_code} - {error_text}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"OpenRouter API error: {response.status_code} - {error_text[:200]}"
                )
            
            result = response.json()
            
            # Check for error in response
            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                print(f"❌ OpenRouter returned error: {error_msg}")
                raise HTTPException(status_code=500, detail=f"OpenRouter error: {error_msg}")
            
            # Parse the AI response
            ai_content = result["choices"][0]["message"]["content"]
            parsed_content = json.loads(ai_content)
            
            print(f"✓ OpenRouter API call successful")
            return parsed_content
            
        except httpx.HTTPError as e:
            print(f"❌ HTTP error calling OpenRouter: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"❌ Failed to parse OpenRouter response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

async def call_gemini(transcript_text: str, model: str = "gemini-2.0-flash-exp", video_duration: str = None, video_duration_seconds: float = None) -> dict:
    """Call Google Gemini API to generate chapters and summary"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Create Gemini client
        client = genai.Client(api_key=api_key)
        
        duration_info = f"\nVideo Duration: {video_duration} (max {int(video_duration_seconds)} seconds)" if video_duration else ""
        
        prompt = f"""You are a helpful assistant that analyzes YouTube video transcripts and creates structured chapters with summaries.
{duration_info}

Given the following video transcript with timestamps in [MM:SS] or [HH:MM:SS] format, please:
1. Identify major topic changes and create 5-8 chapters
2. For each chapter, YOU MUST extract the EXACT timestamp_seconds from the [HH:MM:SS] or [MM:SS] markers in the transcript
3. Parse the timestamps like this: [00:45] = 45 seconds, [02:30] = 150 seconds, [01:15:20] = 4520 seconds
4. CRITICAL: All timestamp_seconds MUST be between 0 and {int(video_duration_seconds)} (the video duration)
5. Use timestamps that actually appear in the transcript - do NOT make up timestamps
6. Provide a descriptive title and brief summary (2-3 sentences) for each chapter
7. Create an overall video summary (3-4 sentences)

Transcript:
{transcript_text}

Please respond in the following JSON format:
{{
    "chapters": [
        {{
            "timestamp_seconds": 0,
            "title": "Chapter Title",
            "summary": "Brief summary of this chapter"
        }}
    ],
    "overall_summary": "Overall video summary"
}}"""
        
        # Generate content with proper model name
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse response
        parsed_content = json.loads(response.text)
        return parsed_content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

# Routes
@app.get("/")
async def root():
    return {
        "message": "YouTube Transcript & Chapter Generator API",
        "endpoints": {
            "POST /analyze": "Analyze a YouTube video and generate chapters",
            "GET /transcript/{video_id}": "Get transcript only for a video"
        }
    }

@app.post("/analyze", response_model=VideoResponse)
async def analyze_video(request: VideoRequest):
    """
    Analyze a YouTube video: fetch transcript and generate chapters with AI
    Auto-fallback: Tries Gemini first, falls back to OpenRouter if quota exceeded
    """
    try:
        # Extract video ID
        video_id = extract_video_id(request.video_url)
        
        # Fetch transcript
        try:
            # Try to fetch transcript in specified languages or any available language
            if request.languages:
                # User specified languages
                fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=request.languages)
            else:
                # Try common languages including Hindi, English, Spanish, etc.
                try:
                    fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                except NoTranscriptFound:
                    # If English not found, try to get any available transcript
                    transcript_list = YouTubeTranscriptApi().list(video_id)
                    # Get the first available transcript
                    if transcript_list:
                        first_transcript = next(iter(transcript_list), None)
                        if first_transcript:
                            fetched_transcript = first_transcript.fetch()
                        else:
                            raise NoTranscriptFound(video_id)
                    else:
                        raise NoTranscriptFound(video_id)
        except TranscriptsDisabled:
            raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
        except NoTranscriptFound:
            raise HTTPException(status_code=404, detail="No transcript found for this video in any language")
        
        # Format transcript for AI
        transcript_text = "\n".join([
            f"[{format_timestamp(snippet.start)}] {snippet.text}"
            for snippet in fetched_transcript.snippets
        ])
        
        # Get video duration from last timestamp
        video_duration_seconds = fetched_transcript.snippets[-1].start if fetched_transcript.snippets else 0
        video_duration_formatted = format_timestamp(video_duration_seconds)
        
        # Get AI analysis with automatic fallback
        ai_response = None
        used_provider = request.api_provider or "gemini"
        
        # Try Gemini first (if requested or as default)
        if used_provider == "gemini":
            try:
                ai_response = await call_gemini(
                    transcript_text, 
                    request.model or "gemini-2.0-flash-exp",
                    video_duration_formatted,
                    video_duration_seconds
                )
            except HTTPException as e:
                # Check if it's a quota/rate limit error (429)
                if "429" in str(e.detail) or "RESOURCE_EXHAUSTED" in str(e.detail) or "quota" in str(e.detail).lower():
                    print(f"⚠️  Gemini quota exceeded, falling back to OpenRouter...")
                    # Check if OpenRouter is available
                    if os.getenv("OPENROUTER_API_KEY"):
                        try:
                            ai_response = await call_openrouter(
                                transcript_text, 
                                "anthropic/claude-3-haiku",
                                video_duration_formatted,
                                video_duration_seconds
                            )
                            used_provider = "openrouter (fallback)"
                            print(f"✓ Successfully used OpenRouter as fallback")
                        except Exception as fallback_error:
                            raise HTTPException(
                                status_code=503, 
                                detail=f"Gemini quota exceeded and OpenRouter fallback failed. Please try again later or configure OPENROUTER_API_KEY. Error: {str(fallback_error)}"
                            )
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="Gemini quota exceeded. Please configure OPENROUTER_API_KEY in .env for automatic fallback, or wait for quota reset."
                        )
                else:
                    # Re-raise if it's not a quota error
                    raise
        else:
            # Use OpenRouter directly if requested
            ai_response = await call_openrouter(
                transcript_text, 
                request.model or "anthropic/claude-3-haiku",
                video_duration_formatted,
                video_duration_seconds
            )
        
        # Format chapters
        chapters = [
            Chapter(
                timestamp=format_timestamp(ch["timestamp_seconds"]),
                title=ch["title"],
                summary=ch["summary"]
            )
            for ch in ai_response.get("chapters", [])
        ]
        
        # Format transcript segments
        transcript_segments = [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration
            )
            for snippet in fetched_transcript.snippets
        ]
        
        # Log which provider was used
        print(f"✓ Chapters generated successfully using: {used_provider}")
        
        return VideoResponse(
            video_id=video_id,
            transcript=transcript_segments,
            chapters=chapters,
            summary=ai_response.get("overall_summary", "")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/transcript/{video_id}")
async def get_transcript(video_id: str):
    """
    Get transcript only for a YouTube video
    """
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
        
        transcript_segments = [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration
            )
            for snippet in fetched_transcript.snippets
        ]
        
        return {
            "video_id": video_id,
            "transcript": transcript_segments
        }
        
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/ai-question")
async def answer_question(request: AIQuestionRequest):
    """
    Answer student questions about a video using AI based on the video transcript
    """
    try:
        # Extract video ID and fetch transcript
        video_id = extract_video_id(request.video_url)
        
        try:
            fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        except NoTranscriptFound:
            # Try to get any available transcript
            transcript_list = YouTubeTranscriptApi().list(video_id)
            first_transcript = next(iter(transcript_list), None)
            if first_transcript:
                fetched_transcript = first_transcript.fetch()
            else:
                raise HTTPException(status_code=404, detail="No transcript available for this video")
        except TranscriptsDisabled:
            raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
        
        # Format transcript for AI
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript.snippets])
        
        # Limit transcript length to avoid token limits (use first ~10000 characters)
        if len(transcript_text) > 10000:
            transcript_text = transcript_text[:10000] + "..."
        
        prompt = f"""You are an educational assistant helping students understand video content.

Video Title: {request.video_title}

Video Transcript:
{transcript_text}

Student Question: {request.question}

Based on the video transcript above, provide a detailed answer to the student's question. Include:
- Direct references to what was said in the video
- Relevant concepts and definitions from the transcript
- Formulas or steps mentioned (if applicable)
- Examples from the video content
- Clear explanations with headings and bullet points

If the question cannot be answered from the transcript, politely explain that the information is not covered in this video."""

        # Try Gemini first, fallback to OpenRouter on quota error
        ai_response = None
        used_provider = request.api_provider or "gemini"
        
        if used_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="Gemini API key not configured")
            
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt
                )
                ai_response = response.text
            except Exception as e:
                # Check if it's a quota error
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"⚠️  Gemini quota exceeded for AI question, falling back to OpenRouter...")
                    # Fallback to OpenRouter
                    if os.getenv("OPENROUTER_API_KEY"):
                        try:
                            api_key = os.getenv("OPENROUTER_API_KEY")
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                response = await client.post(
                                    "https://openrouter.ai/api/v1/chat/completions",
                                    headers={
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type": "application/json"
                                    },
                                    json={
                                        "model": "anthropic/claude-3-haiku",
                                        "messages": [{"role": "user", "content": prompt}]
                                    }
                                )
                                
                                if response.is_success:
                                    result = response.json()
                                    ai_response = result["choices"][0]["message"]["content"]
                                    used_provider = "openrouter (fallback)"
                                    print(f"✓ Successfully used OpenRouter as fallback for AI question")
                                else:
                                    raise HTTPException(status_code=500, detail=f"OpenRouter fallback failed: {response.status_code}")
                        except Exception as fallback_error:
                            raise HTTPException(
                                status_code=503,
                                detail=f"Gemini quota exceeded and OpenRouter fallback failed: {str(fallback_error)}"
                            )
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="Gemini quota exceeded. Please configure OPENROUTER_API_KEY for automatic fallback."
                        )
                else:
                    # Re-raise if not a quota error
                    raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")
        else:
            # Use OpenRouter directly if specified
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "anthropic/claude-3-haiku",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                
                if not response.is_success:
                    raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.status_code}")
                
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
        
        # Return the response
        print(f"✓ AI question answered using: {used_provider}")
        return {"answer": ai_response}
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

@app.post("/generate-quiz")
async def generate_quiz(request: QuizRequest):
    """
    Generate a quiz based on actual video transcript content
    """
    try:
        # Extract video ID and fetch transcript
        video_id = extract_video_id(request.video_url)
        
        try:
            fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        except NoTranscriptFound:
            # Try to get any available transcript
            transcript_list = YouTubeTranscriptApi().list(video_id)
            first_transcript = next(iter(transcript_list), None)
            if first_transcript:
                fetched_transcript = first_transcript.fetch()
            else:
                raise HTTPException(status_code=404, detail="No transcript available for this video")
        except TranscriptsDisabled:
            raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
        
        # Format transcript for AI
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript.snippets])
        
        # Limit transcript length to avoid token limits
        if len(transcript_text) > 8000:
            transcript_text = transcript_text[:8000] + "..."
        
        prompt = f"""Based on the following video transcript, generate a quiz with 5 multiple-choice questions.

Video Title: {request.video_title}

Video Transcript:
{transcript_text}

Create questions that:
- Test understanding of KEY CONCEPTS actually discussed in the video
- Cover different parts of the video content
- Have 4 options (A, B, C, D) where only one is correct
- Include the correct answer

Return ONLY a JSON array with this exact structure (no additional text):
[
  {{
    "question": "What is...",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "correct": "A"
  }}
]"""

        # Try Gemini first, fallback to OpenRouter on quota error
        quiz_data = None
        used_provider = request.api_provider or "gemini"
        
        if used_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="Gemini API key not configured")
            
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                quiz_data = json.loads(response.text)
            except Exception as e:
                # Check if it's a quota error
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"⚠️  Gemini quota exceeded for quiz generation, falling back to OpenRouter...")
                    # Fallback to OpenRouter
                    if os.getenv("OPENROUTER_API_KEY"):
                        try:
                            api_key = os.getenv("OPENROUTER_API_KEY")
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                response = await client.post(
                                    "https://openrouter.ai/api/v1/chat/completions",
                                    headers={
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type": "application/json"
                                    },
                                    json={
                                        "model": "anthropic/claude-3-haiku",
                                        "messages": [{"role": "user", "content": prompt}],
                                        "response_format": {"type": "json_object"}
                                    }
                                )
                                
                                if response.is_success:
                                    result = response.json()
                                    quiz_text = result["choices"][0]["message"]["content"]
                                    quiz_data = json.loads(quiz_text)
                                    used_provider = "openrouter (fallback)"
                                    print(f"✓ Successfully used OpenRouter as fallback for quiz generation")
                                else:
                                    raise HTTPException(status_code=500, detail=f"OpenRouter fallback failed: {response.status_code}")
                        except Exception as fallback_error:
                            raise HTTPException(
                                status_code=503,
                                detail=f"Gemini quota exceeded and OpenRouter fallback failed: {str(fallback_error)}"
                            )
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="Gemini quota exceeded. Please configure OPENROUTER_API_KEY for automatic fallback."
                        )
                else:
                    # Re-raise if not a quota error
                    raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")
        else:
            # Use OpenRouter directly if specified
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "anthropic/claude-3-haiku",
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if not response.is_success:
                    raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.status_code}")
                
                result = response.json()
                quiz_text = result["choices"][0]["message"]["content"]
                quiz_data = json.loads(quiz_text)
        
        # Return the quiz
        print(f"✓ Quiz generated using: {used_provider}")
        return {"quiz": quiz_data}
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
