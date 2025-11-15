# videoaudio-transcode

Video transcoding utility that extracts audio from videos, transcribes it to text using local AI models, and formats the output as Markdown.

## Overview

This tool walks through a directory tree and processes video files automatically:

1. **Extract Audio** - Extracts audio track from videos and saves as AAC format (optimized for voice)
2. **Transcribe** - Uses OpenAI Whisper (local) to convert speech to text
3. **Format** - Creates a formatted Markdown document using Ollama LLM or simple text processing

### Supported Video Formats

- `.mp4` - MPEG-4 Video
- `.mkv` - Matroska Video
- `.avi` - Audio Video Interleave
- `.mov` - QuickTime Movie

### Output Files

For each video file (e.g., `lecture.mp4`), the script generates:

- `lecture.aac` - Extracted audio (AAC format, optimized for voice)
- `lecture.txt` - Raw transcription text
- `lecture.md` - Formatted Markdown document

---

## Installation

### Quick Install

Run the automated installation script:

```bash
chmod +x install.sh
./install.sh
```

This will install:
- FFmpeg (audio/video processing)
- Python 3 and pip
- OpenAI Whisper (local speech-to-text Python package)
- Ollama (LLM for text formatting)
- jq (JSON processing)
- Basic utilities (sed, awk, fmt)

### Manual Installation (Ubuntu/Debian)

#### 1. Install FFmpeg
```bash
sudo apt update
sudo apt install -y ffmpeg
```

#### 2. Install Python 3 and pip
```bash
sudo apt install -y python3 python3-pip python3-venv python3-full
```

#### 3. Install OpenAI Whisper (Python package)

**Recommended method (using pipx for isolated installation):**
```bash
# Install pipx
sudo apt install -y pipx
pipx ensurepath

# Install whisper
pipx install openai-whisper

# Reload shell configuration
source ~/.bashrc
```

**Alternative method (if you prefer pip):**
```bash
# Install in user directory (may conflict with system Python on newer Ubuntu)
python3 -m pip install --user --upgrade openai-whisper

# Or create a virtual environment
python3 -m venv ~/whisper-env
source ~/whisper-env/bin/activate
pip install openai-whisper
```

**Note:** On Ubuntu 23.04+ and Debian 12+, the system Python is externally managed (PEP 668). Using `pipx` is the recommended approach.

#### 4. Install Ollama via SNAP
```bash
sudo snap install ollama
```

After installation, pull a model for text formatting:
```bash
ollama pull llama2
# or
ollama pull mistral
```

#### 5. Install jq for JSON processing
```bash
sudo apt install -y jq
```

---

## Usage

### Basic Usage

```bash
chmod +x transcode.sh
./transcode.sh [OPTIONS] <directory>
```

### Examples

**Process videos in current directory:**
```bash
./transcode.sh .
```

**Process videos in specific folder:**
```bash
./transcode.sh /path/to/videos
```

**Use GPU acceleration (requires CUDA):**
```bash
./transcode.sh --gpu /path/to/videos
```

**Use larger Whisper model for better accuracy:**
```bash
./transcode.sh --transcription-model large-v3 /path/to/videos
```

**Use specific Ollama model:**
```bash
./transcode.sh --format-model mistral /path/to/videos
```

**Use simple text formatting (no LLM, faster):**
```bash
./transcode.sh --simple-format /path/to/videos
```

**Verbose output for debugging:**
```bash
./transcode.sh --verbose /path/to/videos
```

**Combined options:**
```bash
./transcode.sh --gpu --transcription-model large-v3 --format-model mistral /path/to/videos
```

---

## Command Line Options

```
-g, --gpu                       Use GPU acceleration (default: CPU)
-c, --cpu                       Force CPU usage
-t, --transcription-model       Whisper model to use (default: base)
                                Options: tiny, base, small, medium, large, large-v2, large-v3
-f, --format-model              Ollama model for text formatting (default: llama2)
-s, --simple-format             Use simple bash-based formatting instead of LLM
-v, --verbose                   Enable verbose output
-h, --help                      Display help message
```

---

## Whisper Models

Whisper models are downloaded automatically on first use to `~/.cache/whisper/`.

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny` | ~1 GB | Fastest | Lowest | Quick tests |
| `base` | ~1 GB | Fast | Good | **Default, balanced** |
| `small` | ~2 GB | Medium | Better | Good quality |
| `medium` | ~5 GB | Slow | High | High quality |
| `large-v3` | ~3 GB | Slowest | Best | **Maximum accuracy** |

### Pre-download Models (Optional)

To download a model before processing:

```bash
python3 -c "import whisper; whisper.load_model('large-v3')"
```

---

## Ollama Models

### Install Additional Models

```bash
# List available models
ollama list

# Pull additional models
ollama pull mistral
ollama pull llama2
ollama pull codellama
```

Popular models for text formatting:
- `llama2` - Default, good balance
- `mistral` - Fast and accurate
- `llama3` - Latest, best quality

---

## Troubleshooting

### Python package not found
```bash
# Recommended: Install with pipx
sudo apt install -y pipx
pipx install openai-whisper
source ~/.bashrc

# Alternative: Install with pip in user directory
python3 -m pip install --user --upgrade openai-whisper
```

### Whisper import error in script
If you get "ModuleNotFoundError: No module named 'whisper'":
```bash
# Check if whisper is installed
python3 -c "import whisper; print('Whisper OK')"

# If using pipx, ensure PATH is updated
source ~/.bashrc

# Reinstall if needed
pipx reinstall openai-whisper
```

### Whisper model download issues
Models download automatically to `~/.cache/whisper/`. If download fails:
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper/
# Run transcription again - it will re-download
```

### Ollama not running
```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### Check dependencies
```bash
which ffmpeg
which python3
python3 -c "import whisper; print('Whisper OK')"
which ollama
which jq
```

### Permission denied
```bash
chmod +x install.sh transcode.sh whisper_transcribe.py
```

### No audio track found
The script automatically skips videos without audio tracks. Check with:
```bash
ffprobe -v error -select_streams a -show_entries stream=codec_type video.mp4
```

---

## Performance Tips

- **GPU Acceleration**: Use `--gpu` flag if you have NVIDIA GPU with CUDA support
- **Model Selection**: Use smaller models (`tiny`, `base`) for faster processing
- **Simple Formatting**: Use `--simple-format` to skip LLM formatting (much faster)
- **Batch Processing**: Process multiple videos in one directory for better efficiency
- **Model Persistence**: First run downloads models; subsequent runs are faster

---

## How It Works

1. **Directory Scanning** - Recursively finds all video files with supported extensions
2. **Audio Detection** - Checks if video has an audio track using FFprobe
3. **Audio Extraction** - Extracts audio using FFmpeg with AAC codec optimized for voice (mono, 16kHz, 64kbps)
4. **Transcription** - Processes audio through Whisper model running locally
5. **Formatting** - Formats raw transcript into structured Markdown using:
   - **LLM mode**: Ollama with chosen model for intelligent formatting
   - **Simple mode**: Bash utilities (sed, awk, fmt) for basic paragraph formatting
6. **Output** - Saves all files (audio, text, markdown) alongside original video

---

## Project Structure

```
videoaudio-transcode/
├── install.sh              # Installation script
├── transcode.sh            # Main transcoding script (bash)
├── whisper_transcribe.py   # Python wrapper for Whisper
└── README.md               # This file
```

---

## Requirements

- **OS**: Ubuntu/Debian Linux (tested on Ubuntu 20.04+)
- **Disk Space**: ~5-10 GB for models and dependencies
- **RAM**: 4GB minimum, 8GB+ recommended for larger models
- **GPU**: Optional, NVIDIA GPU with CUDA for faster processing

---

## License

This project is provided as-is for educational and personal use.



