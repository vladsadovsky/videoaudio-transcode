# Video/Audio Transcoding & Transcription

A tool for extracting audio from videos, transcribing with OpenAI Whisper, and formatting transcriptions to Markdown using LLMs.

## Project Structure

```
.
├── bash-version/          # Original bash script implementation
│   ├── transcode.sh       # Main bash script
│   ├── whisper_transcribe.py  # Python Whisper wrapper
│   ├── install.sh         # Installation script
│   └── README.md          # Bash version documentation
│
├── transcode.py           # ⭐ Python version (recommended)
├── install.sh             # Python version installer
└── README_PYTHON.md       # Python version documentation
```

## Quick Start

### Python Version (Recommended)

```bash
# Install dependencies (auto-downloads optimal models for your GPU)
./install.sh

# Process videos (auto-detects GPU and selects best model)
./transcode.py /path/to/videos

# Get help
./transcode.py --help
```

**For model selection guide, see [MODELS.md](MODELS.md)**

See [README_PYTHON.md](README_PYTHON.md) for detailed documentation.

See [README_PYTHON.md](README_PYTHON.md) for detailed documentation.

### Bash Version

```bash
cd bash-version
./install.sh
./transcode.sh /path/to/videos
```

See [bash-version/README.md](bash-version/README.md) for documentation.

## Which Version to Use?

**Python version** is recommended for:
- ✅ Better error messages and diagnostics
- ✅ Easier maintenance and updates
- ✅ Cleaner codebase
- ✅ More detailed progress tracking

**Bash version** if you:
- Need to work on systems without Python 3.7+
- Prefer shell scripts
- Have existing workflows integrated with the bash version

Both versions have **identical CLI interfaces** and produce the same output

## Features

- Extract audio from video files (MP4, MKV, AVI, MOV)
- Transcribe audio using OpenAI Whisper
- Format transcriptions to Markdown using LLMs
- GPU acceleration support
- Multiple model options
- Batch processing

## License

See LICENSE file for details.
