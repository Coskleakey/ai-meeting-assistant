# 🎙️ AI Meeting Assistant

A fully offline AI-powered meeting assistant that transcribes audio recordings and extracts actionable insights using local LLMs.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎤 **Audio Transcription**: Convert audio/video to text using faster-whisper
- 🤖 **AI Analysis**: Extract meeting summary and action items using Ollama
- 📊 **Analytics Dashboard**: Interactive charts showing task distribution and insights
- 💾 **Multiple Outputs**: Save as JSON, CSV, and TXT
- 🖥️ **Dual Interface**: Command-line tool and Streamlit web UI
- 🔒 **100% Offline**: All processing happens locally (no API keys needed)

## 🎯 Demo

### Streamlit Dashboard
![Dashboard Preview](screenshots/dashboard.png)

### Analytics View
![Analytics Preview](screenshots/analytics.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed and running
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-meeting-assistant.git
cd ai-meeting-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Pull the Ollama model**
```bash
ollama pull llama3.2
```

### Usage

#### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Option 2: Command Line
```bash
# Analyze an audio file
python meeting_analyzer.py meeting_recording.mp3

# Analyze a text transcript
python meeting_analyzer.py transcript.txt
```

## 📁 Project Structure
```
ai-meeting-assistant/
├── app.py                  # Streamlit web interface
├── meeting_analyzer.py     # Core transcription & analysis functions
├── main.py                 # CLI script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
├── meeting_outputs/       # Output directory (auto-created)
└── temp_uploads/          # Temporary upload storage (auto-created)
```

## 🎨 Features in Detail

### Transcription
- Supports multiple audio formats: MP3, WAV, M4A, OGG, FLAC, AAC, WMA
- Supports video formats: MP4, AVI, MKV (extracts audio)
- Uses faster-whisper for efficient CPU transcription
- Voice Activity Detection (VAD) to filter silence

### AI Analysis
- Extracts meeting summary (2-4 sentences)
- Identifies action items with owners and deadlines
- Runs 100% locally using Ollama
- Structured JSON output

### Analytics Dashboard
- Task distribution by team member
- Deadline urgency timeline (Overdue, This Week, Next Week, Later)
- Top keywords from meeting
- Detailed task breakdown by owner

## 📊 Output Formats

The tool generates three output files:

1. **Transcript (.txt)**: Full meeting transcription
2. **JSON (.json)**: Structured data with summary and tasks
3. **CSV (.csv)**: Task list for spreadsheet import

Example JSON output:
```json
{
  "summary": "Team discussed Q1 progress with 15% revenue growth...",
  "tasks": [
    {
      "task": "Fix critical bugs in payment flow",
      "owner": "Sarah",
      "deadline": "End of this week"
    }
  ]
}
```

## 🛠️ Configuration

### Whisper Model Selection

Choose model size in Streamlit sidebar or via code:
- `tiny`: Fastest, least accurate
- `base`: Balanced (default)
- `small`: More accurate, slower
- `medium`: Most accurate, slowest

### Ollama Model

Default: `llama3.2`

To use a different model, edit `meeting_analyzer.py`:
```python
response = ollama.chat(
    model='llama3.2',  # Change this
    ...
)
```

## 🧪 Example Usage
```python
from meeting_analyzer import transcribe_audio, analyze_meeting

# Transcribe audio
transcript = transcribe_audio("meeting.mp3", model_size="base")

# Analyze with AI
result = analyze_meeting(transcript)

print(result['summary'])
for task in result['tasks']:
    print(f"- {task['task']} ({task['owner']}, {task['deadline']})")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast audio transcription
- [Ollama](https://ollama.com) - Local LLM runtime
- [Streamlit](https://streamlit.io) - Web interface framework

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/ai-meeting-assistant](https://github.com/yourusername/ai-meeting-assistant)

## 🎯 Roadmap

- [ ] Speaker diarization (identify who said what)
- [ ] Support for multiple languages
- [ ] Export to Notion, Trello, Asana
- [ ] Real-time transcription
- [ ] Meeting recording directly in the app
- [ ] Docker containerization

## ⚠️ Known Issues

- Very long meetings (>2 hours) may require model with larger context window
- GPU support requires CUDA installation
- First run downloads Whisper model (~150MB for base model)

## 💡 Tips

- Use "base" model for best speed/accuracy balance
- Ensure Ollama is running before starting the app
- For best results, use high-quality audio recordings
- Meeting recordings should have clear speech