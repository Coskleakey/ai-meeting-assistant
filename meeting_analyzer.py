import ollama
import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import sys
from datetime import datetime
from faster_whisper import WhisperModel


def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio file to text using faster-whisper.
    
    Args:
        file_path: Path to audio file (mp3, wav, m4a, etc.)
        model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
                   Default is 'base' for speed/accuracy balance
        
    Returns:
        Transcribed text as a single string
    """
    print(f"\n🎤 Transcribing audio with faster-whisper ({model_size} model)...")
    
    try:
        # Initialize the model
        # device="cpu" for CPU, "cuda" for GPU
        # compute_type="int8" for CPU, "float16" for GPU
        model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8"
        )
        
        print(f"   Loading audio file: {file_path}")
        
        # Transcribe the audio
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language="en",  # Set to None for auto-detection
            vad_filter=True,  # Voice Activity Detection to filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )
        
        print(f"   Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"   Duration: {info.duration:.2f} seconds")
        print("   Transcribing segments...")
        
        # Collect all segments into a single transcript
        # IMPORTANT: Process ALL segments, not just the first few
        transcript_parts = []
        segment_count = 0
        
        for segment in segments:
            transcript_parts.append(segment.text.strip())
            segment_count += 1
            
            # Print progress every 10 segments to avoid spam
            if segment_count % 10 == 0:
                print(f"      Processed {segment_count} segments... (up to {segment.end:.1f}s)")
        
        transcript = " ".join(transcript_parts)
        
        print(f"   Total segments processed: {segment_count}")
        print(f"✓ Transcription complete ({len(transcript)} characters, {len(transcript.split())} words)")
        
        return transcript
        
    except FileNotFoundError:
        print(f"Error: Audio file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def analyze_meeting(transcript: str) -> Dict[str, Any]:
    """
    Analyze a meeting transcript using a local Ollama LLM to extract
    summary and action items.
    
    Args:
        transcript: The meeting transcript as a string
        
    Returns:
        Dictionary with 'summary' and 'tasks' keys in structured format
    """
    
    system_prompt = """You are an expert meeting analyst. Your task is to analyze meeting transcripts and extract:
1. A concise summary of the meeting (2-4 sentences)
2. All action items/tasks mentioned, including who is responsible and any deadlines

For each task, identify:
- The specific task or action item
- The owner/person responsible (use "Unassigned" if not mentioned)
- The deadline (use "Not specified" if not mentioned, but infer from context like "by next week", "end of month", etc.)

You MUST respond with valid JSON only, no other text. Use this exact format:
{
  "summary": "Brief summary of the meeting discussion and key decisions",
  "tasks": [
    {
      "task": "Specific action item description",
      "owner": "Person's name or Unassigned",
      "deadline": "Specific date or timeframe or Not specified"
    }
  ]
}

Be thorough - extract ALL tasks mentioned, even if briefly referenced. If no tasks were identified, return an empty tasks array."""

    try:
        print(f"   Analyzing transcript ({len(transcript)} characters, {len(transcript.split())} words)...")
        
        # Call the local Ollama model
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': f"Please analyze this meeting transcript and extract the summary and action items:\n\n{transcript}"
                }
            ],
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_ctx': 8192,  # Increase context window for longer transcripts
            }
        )
        
        # Extract the response content
        response_text = response['message']['content']
        
        # Parse the JSON response
        response_text = response_text.strip()
        
        # Find JSON object boundaries
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            result = {
                "summary": "Error: Could not parse LLM response",
                "tasks": []
            }
        
        # Validate structure
        if 'summary' not in result:
            result['summary'] = "Summary not provided"
        if 'tasks' not in result:
            result['tasks'] = []
        
        # Ensure each task has required fields
        for task in result['tasks']:
            if 'task' not in task:
                task['task'] = "Task description missing"
            if 'owner' not in task:
                task['owner'] = "Unassigned"
            if 'deadline' not in task:
                task['deadline'] = "Not specified"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_text}")
        return {
            "summary": "Error: Invalid JSON response from model",
            "tasks": []
        }
    except Exception as e:
        print(f"Error analyzing meeting: {e}")
        import traceback
        traceback.print_exc()
        return {
            "summary": f"Error: {str(e)}",
            "tasks": []
        }


def read_transcript(file_path: str) -> str:
    """
    Read transcript from a text file.
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        Transcript content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def save_transcript(transcript: str, output_path: str) -> None:
    """
    Save transcript to a text file.
    
    Args:
        transcript: Transcript text
        output_path: Path for output text file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"✓ Transcript saved to: {output_path}")
    except Exception as e:
        print(f"Error saving transcript: {e}")


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis results as JSON file.
    
    Args:
        data: Analysis results dictionary
        output_path: Path for output JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON saved to: {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")


def save_csv(data: Dict[str, Any], output_path: str) -> None:
    """
    Save tasks as CSV file using pandas.
    
    Args:
        data: Analysis results dictionary
        output_path: Path for output CSV file
    """
    try:
        if data['tasks']:
            df = pd.DataFrame(data['tasks'])
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ CSV saved to: {output_path}")
        else:
            # Create empty CSV with headers
            df = pd.DataFrame(columns=['task', 'owner', 'deadline'])
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ CSV saved to: {output_path} (no tasks found)")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def print_formatted_results(data: Dict[str, Any]) -> None:
    """
    Print analysis results in a formatted, readable way.
    
    Args:
        data: Analysis results dictionary
    """
    print("\n" + "="*70)
    print("MEETING ANALYSIS RESULTS")
    print("="*70)
    
    print("\n📋 SUMMARY:")
    print("-" * 70)
    print(data['summary'])
    
    print("\n\n✅ ACTION ITEMS:")
    print("-" * 70)
    
    if data['tasks']:
        for i, task in enumerate(data['tasks'], 1):
            print(f"\n{i}. Task: {task['task']}")
            print(f"   Owner: {task['owner']}")
            print(f"   Deadline: {task['deadline']}")
    else:
        print("\nNo action items identified.")
    
    print("\n" + "="*70)
    print(f"Total Tasks: {len(data['tasks'])}")
    print("="*70 + "\n")


def is_audio_file(file_path: str) -> bool:
    """
    Check if the file is an audio file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if audio file, False otherwise
    """
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.mp4', '.avi', '.mkv'}
    return Path(file_path).suffix.lower() in audio_extensions


def main():
    """
    Main function to orchestrate the meeting analysis workflow.
    Supports both audio files (transcription) and text files (direct processing).
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python meeting_analyzer.py <audio_or_transcript_file>")
        print("\nSupported formats:")
        print("  Audio: mp3, wav, m4a, ogg, flac, aac, etc.")
        print("  Text: txt, text")
        print("\nExamples:")
        print("  python meeting_analyzer.py meeting.mp3")
        print("  python meeting_analyzer.py transcript.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Create output directory if it doesn't exist
    output_dir = Path("meeting_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file).stem
    transcript_output = output_dir / f"{base_name}_{timestamp}_transcript.txt"
    json_output = output_dir / f"{base_name}_{timestamp}.json"
    csv_output = output_dir / f"{base_name}_{timestamp}.csv"
    
    print("\n" + "="*70)
    print("OFFLINE MEETING ASSISTANT")
    print("="*70)
    print(f"\nInput file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Get transcript (either from audio or text file)
    if is_audio_file(input_file):
        # Transcribe audio file
        transcript = transcribe_audio(input_file)
        
        # Save the transcript
        print("\n💾 Saving transcript...")
        save_transcript(transcript, str(transcript_output))
    else:
        # Read text transcript
        print("\n📖 Reading transcript...")
        transcript = read_transcript(input_file)
        print(f"✓ Transcript loaded ({len(transcript)} characters)")
    
    # Step 2: Analyze meeting
    print("\n🤖 Analyzing meeting with Ollama (llama3.2)...")
    print("   This may take a moment...")
    result = analyze_meeting(transcript)
    print("✓ Analysis complete")
    
    # Step 3: Save outputs
    print("\n💾 Saving outputs...")
    save_json(result, str(json_output))
    save_csv(result, str(csv_output))
    
    # Step 4: Print formatted results
    print_formatted_results(result)
    
    print("✨ Process complete!\n")


if __name__ == "__main__":
    main()