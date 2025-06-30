import os
import hashlib
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import ffmpeg
from faster_whisper import WhisperModel
import google.generativeai as genai


# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# --- Directory Setup ---
VIDEOS_DIR = Path("videos")
TRANSCRIPTS_DIR = Path("transcripts")
VIDEOS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# --- Model Loading ---
MODEL_SIZE = "base"
whisper_model = WhisperModel(MODEL_SIZE , device="cpu", compute_type="int8")

def get_video_hash(file_content: bytes) -> str:
    """Calculates the SHA256 hash of the video file."""
    return hashlib.sha256(file_content).hexdigest()

@app.post("/generate-mcqs/")
async def generate_mcqs_from_video(video: UploadFile = File(...)):
    """
    Accepts a video file, transcribes it, and generates MCQs.
    """
    video_content = await video.read()
    video_hash = get_video_hash(video_content)
    transcript_path = TRANSCRIPTS_DIR / f"{video_hash}.json"

    if transcript_path.exists():
        print("loading existing transcript")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
    else:
        print("generating video transcript and MCQS")
        # Save the video temporarily
        video_path = VIDEOS_DIR / f"{video_hash}_{video.filename}"
        with open(video_path, "wb") as f:
            f.write(video_content)

        try:
            # 1. Split video into 5-minute segments and extract audio
            segment_duration = 300  # 5 minutes in seconds
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe["format"]["duration"])
            num_segments = int(duration // segment_duration) + 1

            audio_segments = []
            for i in range(num_segments):
                print(f"Processing segment {i + 1}/{num_segments}...")
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                audio_segment_path = VIDEOS_DIR / f"{video_hash}_segment_{i}.mp3"

                (
                    ffmpeg
                    .input(str(video_path), ss=start_time, t=end_time - start_time)
                    .output(str(audio_segment_path), acodec='mp3')
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                audio_segments.append(str(audio_segment_path))

            # 2. Transcribe audio segments using faster-whisper
            transcripts = []
            for audio_segment in audio_segments:
                print(f"Transcribing segment: {audio_segment}")
                segments, _ = whisper_model.transcribe(audio_segment)
                segment_transcript = " ".join([segment.text for segment in segments])
                transcripts.append(segment_transcript)
                os.remove(audio_segment) # Clean up audio segments

            # 3. Save the transcript
            with open(transcript_path, "w") as f:
                print(f"Saving transcript to {transcript_path}")
                json.dump(transcripts, f)

        finally:
            # Clean up the uploaded video file
            print(f"Cleaning up video file: {video_path}")
            os.remove(video_path)


    # 4. Call Gemini API to generate MCQs
    mcqs = []
    valid_transcripts = [t for t in transcripts if t.strip()]

    if valid_transcripts:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Using a capable model
        print("Generating MCQs using a single Gemini API call...")

        # Create a single prompt with all transcripts
        prompt_parts = [
            "Based on the following list of transcripts from video segments, please generate one multiple-choice question (MCQ) for each segment.",
            "The output should be a single JSON array, where each element is an object corresponding to a transcript.",
            "Each JSON object must contain these keys: 'segment', 'question', 'options' (which is an object with A, B, C, D), and 'answer'.",
            "The 'segment' number should correspond to the order of the transcript in the input.",
            "\n---TRANSCRIPTS---"
        ]
        for i, transcript in enumerate(valid_transcripts):
            prompt_parts.append(f"\nSegment {i + 1}:\n\"\"\"\n{transcript}\n\"\"\"")
        
        prompt = "\n".join(prompt_parts)

        try:
            response = gemini_model.generate_content(prompt)
            # Clean up the response to extract the JSON part
            response_text = response.text.strip()
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            mcq_json_string = response_text[json_start:json_end]

            generated_mcqs = json.loads(mcq_json_string)
            mcqs = generated_mcqs
            print(f"Successfully generated {len(mcqs)} MCQs.")

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing Gemini API JSON response: {e}")
            print(f"Raw response was:\n{response.text}")
        except Exception as e:
            print(f"An unexpected error occurred during Gemini API call: {e}")

    return JSONResponse(content={"mcqs": mcqs})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)