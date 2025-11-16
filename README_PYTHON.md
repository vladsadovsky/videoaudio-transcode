# Video/Audio Transcoding Script (Python Version)

A Python-based tool for extracting audio from videos, transcribing with OpenAI Whisper, and formatting transcriptions to Markdown using LLMs.

## Features

- Extract audio from video files (MP4, MKV, AVI, MOV)
- Transcribe audio using OpenAI Whisper
- Format transcriptions to Markdown using Ollama LLMs
- GPU acceleration support
- Multiple model options
- Batch processing of entire directory trees
- Progress tracking and detailed error messages
- Fallback to simple formatting if LLM unavailable

## Installation

Run the installation script to install all dependencies:

```bash
./install.sh
```

This will install:
- ffmpeg (audio/video processing)
- Python 3 and pip
- OpenAI Whisper
- PyTorch (with CUDA support if available)
- Ollama (for LLM formatting)
- requests library

## Usage

### Basic Usage

```bash
# Automatic mode - detects GPU and selects best model
./transcode.py /path/to/videos

# Force GPU acceleration
./transcode.py --gpu /path/to/videos

# Force CPU only
./transcode.py --cpu /path/to/videos
```

**Smart Auto-Detection:**
- Automatically uses GPU if CUDA is available
- Auto-selects optimal Ollama model based on available VRAM
- No configuration needed for most users!

### Command-Line Options

```
OPTIONS:
    -g, --gpu                   Force GPU acceleration
    -c, --cpu                   Force CPU usage
                                (default: auto-detect - use GPU if CUDA available)
    -t, --transcription-model   Whisper model to use (default: base)
                                Options: tiny, base, small, medium, large-v3
    -f, --format-model          Ollama model for text formatting
                                (default: auto-select based on GPU VRAM)
                                Specify to override auto-selection
    -s, --simple-format         Use simple formatting instead of LLM
    --skip-transcoding          Skip processing if output files already exist
                                If .txt and .md exist: skip entirely
                                If only .txt exists: skip transcoding, only format
    -v, --verbose               Enable verbose output
    -h, --help                  Display help message
```

### Examples

Process videos (auto-detect GPU and model):
```bash
./transcode.py .
```

Force CPU only:
```bash
./transcode.py --cpu /path/to/videos
```

Use specific model (override auto-selection):
```bash
./transcode.py -f qwen2.5:14b /path/to/videos
```

Use specific Whisper and format models:
```bash
./transcode.py -t large-v3 -f mistral /path/to/videos
```

Use simple formatting (no LLM):
```bash
./transcode.py --simple-format /path/to/videos
```

Only reformat existing transcriptions:
```bash
./transcode.py --skip-transcoding -f mistral /path/to/videos
```

Verbose mode:
```bash
./transcode.py -v --gpu /path/to/videos
```

## Whisper Models

| Model    | Size  | Speed      | Quality    | VRAM    |
|----------|-------|------------|------------|---------|
| tiny     | 39M   | Very Fast  | Low        | ~1 GB   |
| base     | 74M   | Fast       | Good       | ~1 GB   |
| small    | 244M  | Moderate   | Better     | ~2 GB   |
| medium   | 769M  | Slow       | High       | ~5 GB   |
| large-v3 | 1550M | Very Slow  | Highest    | ~10 GB  |

The default `base` model provides a good balance of speed and quality.

## Ollama Models

The script includes **intelligent auto-selection** of the best Ollama model based on your GPU VRAM:

### Auto-Selected Models by VRAM:

| VRAM  | Auto-Selected Model | Size  | Quality    |
|-------|---------------------|-------|------------|
| 4GB   | gemma2:2b          | ~2GB  | Good       |
| 6GB   | phi3:mini          | ~3GB  | Very Good  |
| 8GB   | qwen2.5:7b         | ~5GB  | Excellent  |
| 12GB  | qwen2.5:14b        | ~10GB | Superior   |
| 16GB+ | qwen2.5:32b        | ~20GB | Best       |

The script automatically:
1. Detects your GPU VRAM
2. Selects the best model that fits
3. Downloads it if not already installed
4. Uses it for all formatting tasks

### Manual Model Selection:

Override auto-selection with `-f`:
```bash
./transcode.py --gpu -f mistral:7b /path/to/videos
```

Popular models for formatting:
- `qwen2.5:7b` - **Recommended default** - Best balance
- `qwen2.5:14b` - Higher quality for 12GB+ VRAM
- `llama3.1:8b` - Alternative high-quality option
- `mistral:7b` - Fast and capable
- `phi3:mini` - Efficient for limited VRAM

Pull models manually with:
```bash
ollama pull <model-name>
```

## Pre-Downloading Ollama Models

The installation script (`./install.sh`) automatically downloads optimal models for your GPU during setup. However, you can also manually pre-download models to ensure they're ready before processing.

### Automatic Download During Installation

The installer detects your GPU VRAM and downloads multiple models as fallback options:

| Your VRAM | Models Auto-Downloaded |
|-----------|------------------------|
| 24GB+     | qwen2.5:14b, llama3.1:13b, qwen2.5:7b, mistral |
| 16-24GB   | qwen2.5:14b, qwen2.5:7b, mistral |
| 12-16GB   | qwen2.5:14b, qwen2.5:7b |
| 8-12GB    | qwen2.5:7b, mistral, phi3:mini |
| 6-8GB     | qwen2.5:7b, phi3:mini |
| 4-6GB     | phi3:mini, gemma2:2b |
| <4GB      | gemma2:2b |

### Manual Pre-Download Commands

To manually download models for your VRAM size:

**For 12GB VRAM (your system):**
```bash
# Primary model (will be auto-selected)
ollama pull qwen2.5:14b

# Fallback option (auto-used if 14b fails)
ollama pull qwen2.5:7b

# Optional: Additional fallback
ollama pull mistral
```

**For 8GB VRAM:**
```bash
ollama pull qwen2.5:7b
ollama pull mistral
ollama pull phi3:mini
```

**For 16GB+ VRAM:**
```bash
ollama pull qwen2.5:14b
ollama pull llama3.1:13b
ollama pull qwen2.5:7b
```

**For 4-6GB VRAM:**
```bash
ollama pull phi3:mini
ollama pull gemma2:2b
```

### Check Downloaded Models

View all downloaded models:
```bash
ollama list
```

Check model sizes:
```bash
# Show detailed model information
ollama show qwen2.5:14b
```

### Why Multiple Models?

The script automatically falls back to smaller models if:
- Primary model won't fit in VRAM
- GPU memory is temporarily limited
- Model fails to load for any reason

Having multiple models pre-downloaded ensures smooth operation without download delays during processing.

### Recommended Approach

**Option 1: Let installer handle it (easiest)**
```bash
./install.sh
# Installer auto-downloads optimal models for your GPU
```

**Option 2: Manual download before first use**
```bash
# For 12GB VRAM system
ollama pull qwen2.5:14b  # Primary (takes 5-10 min)
ollama pull qwen2.5:7b   # Fallback (takes 3-5 min)

# Then run processing
./transcode.py /path/to/videos
```

**Option 3: Download on-demand**
```bash
# Just run the script - it will auto-download as needed
./transcode.py /path/to/videos
# First model will download automatically (may take 5-10 min)
```

## Output Files

For each video file (e.g., `video.mp4`), the script generates:
- `video.aac` - Extracted audio (AAC format, optimized for voice)
- `video.txt` - Raw transcription text
- `video.md` - Formatted Markdown document

## Workflow

1. **Find Videos**: Recursively search directory for video files
2. **Extract Audio**: Convert video to AAC audio (optimized for speech)
3. **Transcribe**: Use Whisper to convert audio to text
4. **Format**: Use Ollama LLM to format text into structured Markdown

## GPU Acceleration

To use GPU acceleration:

1. Ensure you have an NVIDIA GPU with CUDA support
2. Install CUDA and cuDNN
3. PyTorch with CUDA will be installed automatically
4. Use the `--gpu` flag when running the script

Check CUDA availability:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Ollama not responding

If you get "Ollama service is not responding":

```bash
# Start Ollama server
ollama serve

# Or on Linux with systemd
sudo systemctl start ollama
```

### Model not found

If Ollama model is not found:

```bash
# List available models
ollama list

# Pull required model
ollama pull llama2
```

### CUDA out of memory

If you get CUDA out of memory errors:
- Use a smaller Whisper model (`-t small` or `-t base`)
- Fall back to CPU mode (remove `--gpu` flag)
- Close other GPU-using applications

### Import errors

If you get Python import errors:

```bash
# Reinstall Python dependencies
python3 -m pip install --upgrade openai-whisper requests torch
```

## Comparison with Bash Version

The Python version maintains 100% CLI compatibility with the bash version but offers:

### Advantages
- ✅ Better error handling and diagnostics
- ✅ Cleaner code structure and maintainability
- ✅ Easier to extend and modify
- ✅ Better progress tracking
- ✅ More detailed verbose output
- ✅ Proper exception handling
- ✅ Cross-platform compatibility

### Same Features
- ✅ Identical command-line interface
- ✅ Same output files and format
- ✅ Same processing workflow
- ✅ Same model support

## Requirements

- Python 3.7+
- ffmpeg
- OpenAI Whisper
- PyTorch
- requests
- Ollama (optional, for LLM formatting)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.
