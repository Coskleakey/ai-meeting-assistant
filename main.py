"""
main.py - Main script for offline meeting assistant
Handles audio transcription, meeting analysis, and output generation
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from meeting_analyzer import transcribe_audio, analyze_meeting


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


def save_json(data: dict, output_path: str) -> None:
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


def save_csv(data: dict, output_path: str) -> None:
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


def print_formatted_results(data: dict) -> None:
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
    audio_extensions = {
        '.mp3', '.wav', '.m4a', '.ogg', '.flac', 
        '.aac', '.wma', '.mp4', '.avi', '.mkv'
    }
    return Path(file_path).suffix.lower() in audio_extensions


def main():
    """
    Main function to orchestrate the meeting analysis workflow.
    
    Workflow:
    1. Accept audio file path from command line
    2. Transcribe audio to text using faster-whisper
    3. Analyze transcript using Ollama LLM
    4. Save outputs (transcript, JSON, CSV)
    5. Print formatted results
    """
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file>")
        print("\nSupported audio formats:")
        print("  mp3, wav, m4a, ogg, flac, aac, wma")
        print("  mp4, avi, mkv (extracts audio)")
        print("\nExample:")
        print("  python main.py meeting_recording.mp3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Validate that input is an audio file
    if not is_audio_file(audio_file):
        print(f"Error: '{audio_file}' does not appear to be an audio file.")
        print("Supported formats: mp3, wav, m4a, ogg, flac, aac, wma, mp4, avi, mkv")
        sys.exit(1)
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: File '{audio_file}' not found.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("meeting_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_file).stem
    
    transcript_output = output_dir / f"{base_name}_{timestamp}_transcript.txt"
    json_output = output_dir / f"{base_name}_{timestamp}.json"
    csv_output = output_dir / f"{base_name}_{timestamp}.csv"
    
    # Print header
    print("\n" + "="*70)
    print("OFFLINE MEETING ASSISTANT")
    print("="*70)
    print(f"\nAudio file: {audio_file}")
    print(f"Output directory: {output_dir}/")
    print(f"Timestamp: {timestamp}")
    
    # Step 1: Transcribe audio to text
    transcript = transcribe_audio(audio_file, model_size="base")
    
    # Step 2: Save transcript
    print("\n💾 Saving transcript...")
    save_transcript(transcript, str(transcript_output))
    
    # Step 3: Analyze meeting with LLM
    print("\n🤖 Analyzing meeting with Ollama (llama3)...")
    print("   This may take a moment...")
    analysis_result = analyze_meeting(transcript)
    print("✓ Analysis complete")
    
    # Step 4: Save JSON output
    print("\n💾 Saving outputs...")
    save_json(analysis_result, str(json_output))
    
    # Step 5: Save CSV output
    save_csv(analysis_result, str(csv_output))
    
    # Step 6: Print formatted results to console
    print_formatted_results(analysis_result)
    
    # Summary
    print("📁 Output Files:")
    print(f"   • Transcript: {transcript_output}")
    print(f"   • JSON: {json_output}")
    print(f"   • CSV: {csv_output}")
    
    print("\n✨ Process complete!\n")


if __name__ == "__main__":
    main()