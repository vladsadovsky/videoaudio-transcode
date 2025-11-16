#!/usr/bin/env python3
"""
Video/Audio Transcoding Script
Extracts audio from videos, transcribes, and formats the text
"""

import sys
import os

# Try to activate virtual environment if it exists
def activate_venv():
    """Activate virtual environment if available"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, '.venv', 'bin', 'python3')
    
    # If running from venv python, we're already in the venv
    if sys.executable == venv_python or '.venv' in sys.executable:
        return
    
    # If venv exists and we're not in it, re-execute with venv python
    if os.path.exists(venv_python) and sys.executable != venv_python:
        os.execv(venv_python, [venv_python] + sys.argv)

activate_venv()
import argparse
import subprocess
import json
import time
import requests
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Statistics
@dataclass
class Stats:
    total_videos: int = 0
    processed_videos: int = 0
    skipped_videos: int = 0
    failed_videos: int = 0

# Configuration
@dataclass
class Config:
    use_gpu: bool = False
    transcription_model: str = "base"
    format_model: str = "llama2"
    use_simple_format: bool = False
    skip_transcoding: bool = False
    batch_mode: bool = False
    verbose: bool = False
    root_dir: str = ""
    available_format_models: List[str] = None  # Will be populated during initialization
    current_format_model: str = None  # Actually working model

# Supported video extensions
VIDEO_EXTENSIONS = ['mp4', 'mkv', 'avi', 'mov']

# Global stats
stats = Stats()
config = Config()

################################################################################
# Helper Functions
################################################################################

def print_status(msg: str) -> None:
    """Print info message in green"""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def print_warning(msg: str) -> None:
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

def print_error(msg: str) -> None:
    """Print error message in red"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_verbose(msg: str) -> None:
    """Print debug message in blue (only if verbose mode enabled)"""
    if config.verbose:
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {msg}")

def show_help() -> str:
    """Return help text"""
    script_name = os.path.basename(sys.argv[0])
    return f"""Video/Audio Transcoding Script

Usage: {script_name} [OPTIONS] <directory>

OPTIONS:
    -g, --gpu                   Force GPU acceleration
    -c, --cpu                   Force CPU usage
                                (default: auto-detect - use GPU if CUDA available)
    -t, --transcription-model   Whisper model to use (default: base)
                                Options: tiny, base, small, medium, large-v3
    -f, --format-model          Ollama model for text formatting
                                (default: auto-select based on GPU VRAM)
                                Specify to override auto-selection
    -s, --simple-format         Use simple bash-based formatting instead of LLM
    --skip-transcoding          Skip processing if output files already exist
                                If .txt and .md exist: skip entirely
                                If only .txt exists: skip transcoding, only format
    --batch-mode                Batch process all videos (load Whisper model once)
                                Recommended for 10+ videos (5-25% faster)
    -v, --verbose               Enable verbose output
    -h, --help                  Display this help message

ARGUMENTS:
    directory                   Root directory to search for video files
                                Use '.' for current directory

EXAMPLES:
    # Process videos in current directory (auto-detect GPU)
    {script_name} .

    # Force GPU acceleration
    {script_name} --gpu /path/to/videos
    
    # Force CPU only
    {script_name} --cpu /path/to/videos
    
    # Batch mode for faster processing of many videos
    {script_name} --batch-mode /path/to/videos

SUPPORTED VIDEO FORMATS:
    .mp4, .mkv, .avi, .mov

WORKFLOW:
    1. Find all video files in directory tree
    2. Auto-select best Ollama model based on GPU VRAM (if not specified)
    3. Extract audio track to AAC format (unless skipped)
    4. Transcribe audio using Whisper (unless skipped)
    5. Format transcription to Markdown (unless .md already exists with --skip-transcoding)

OUTPUT FILES (for video.mp4):
    video.aac  - Extracted audio
    video.txt  - Raw transcription
    video.md   - Formatted Markdown

AUTO MODEL SELECTION:
    The script automatically detects your GPU VRAM and selects the best
    Ollama model that will fit. Recommended models by VRAM:
    - 4GB:  gemma2:2b, phi3:mini
    - 6GB:  qwen2.5:3b, llama3.2:3b
    - 8GB:  qwen2.5:7b, mistral:7b (recommended)
    - 12GB: qwen2.5:14b, llama3.1:13b
    - 16GB+: qwen2.5:32b (highest quality)
"""

################################################################################
# Validation Functions
################################################################################

def get_gpu_memory() -> Optional[int]:
    """Get available GPU VRAM in MB. Returns None if no GPU or can't detect."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get total memory in bytes, convert to MB
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_mb = total_memory // (1024 * 1024)
            print_verbose(f"Detected GPU VRAM: {total_mb} MB ({total_mb/1024:.1f} GB)")
            return total_mb
    except Exception as e:
        print_verbose(f"Could not detect GPU memory: {e}")
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            total_mb = int(result.stdout.strip().split('\n')[0])
            print_verbose(f"Detected GPU VRAM (nvidia-smi): {total_mb} MB ({total_mb/1024:.1f} GB)")
            return total_mb
    except Exception as e:
        print_verbose(f"Could not detect GPU memory via nvidia-smi: {e}")
    
    return None

def select_best_model_for_vram(vram_mb: int) -> str:
    """
    Select the best Ollama model based on available VRAM.
    
    Model sizes (approximate VRAM requirements):
    - 2B models: ~2GB
    - 3B models: ~3GB
    - 7B models: ~5GB
    - 8B models: ~6GB
    - 13B models: ~8GB
    - 14B models: ~10GB
    - 22B models: ~14GB
    
    We leave ~2GB headroom for Whisper when using GPU acceleration
    """
    
    # Define models with their requirements (MB) and quality score
    # Format: (model_name, vram_required_mb, quality_score)
    model_candidates = [
        ('qwen2.5:32b', 20000, 100),       # Highest quality
        ('qwen2.5:14b', 10000, 95),
        ('llama3.1:13b', 8000, 90),
        ('qwen2.5:7b', 5000, 85),          # Best balance
        ('llama3.1:8b', 6000, 82),
        ('mistral:7b', 5000, 80),
        ('llama3.2:3b', 3000, 75),
        ('qwen2.5:3b', 3000, 78),
        ('phi3:medium', 8000, 77),         # 14B
        ('phi3:mini', 3000, 70),           # 3.8B
        ('gemma2:9b', 6500, 76),
        ('gemma2:2b', 2000, 65),
        ('llama3:8b', 6000, 75),
        ('llama2:13b', 8000, 70),
        ('llama2:7b', 5000, 65),
    ]
    
    # Reserve VRAM for Whisper if using GPU
    reserved_vram = 2000 if config.use_gpu else 0
    available_vram = vram_mb - reserved_vram
    
    print_verbose(f"Available VRAM for LLM: {available_vram} MB (reserved {reserved_vram} MB for Whisper)")
    
    # Find best model that fits
    best_model = None
    best_quality = 0
    
    for model_name, required_vram, quality in model_candidates:
        if required_vram <= available_vram and quality > best_quality:
            best_model = model_name
            best_quality = quality
    
    if best_model:
        print_verbose(f"Best model for {vram_mb} MB VRAM: {best_model} (quality score: {best_quality})")
        return best_model
    
    # Fallback to smallest model
    print_verbose(f"VRAM too low ({available_vram} MB), using smallest model")
    return 'gemma2:2b'

def auto_select_and_pull_model() -> Optional[str]:
    """
    Auto-select best model based on GPU VRAM and pull if needed.
    Returns the selected model name or None if failed.
    """
    print_status("Auto-detecting best Ollama model for your GPU...")
    
    # Get GPU memory
    vram_mb = get_gpu_memory()
    
    if vram_mb is None:
        print_warning("Could not detect GPU VRAM, using default model")
        return config.format_model
    
    # Select best model
    best_model = select_best_model_for_vram(vram_mb)
    print_status(f"Selected model: {best_model} (optimal for {vram_mb} MB / {vram_mb/1024:.1f} GB VRAM)")
    
    # Check if model is already installed
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed_models = set()
        for line in result.stdout.split('\n')[1:]:
            if line.strip():
                model_name = line.split()[0]
                installed_models.add(model_name.split(':')[0])
        
        model_base = best_model.split(':')[0]
        
        if model_base in installed_models:
            print_status(f"✓ Model '{best_model}' already installed")
            return best_model
        
        # Model not installed, pull it
        print_status(f"Pulling model '{best_model}' (this may take a few minutes)...")
        pull_result = subprocess.run(
            ['ollama', 'pull', best_model],
            capture_output=False,  # Show progress
            text=True
        )
        
        if pull_result.returncode == 0:
            print_status(f"✓ Model '{best_model}' pulled successfully")
            return best_model
        else:
            print_error(f"Failed to pull model '{best_model}'")
            return None
            
    except Exception as e:
        print_error(f"Failed to auto-select model: {e}")
        return None

def check_dependencies() -> None:
    """Check if all required dependencies are installed"""
    missing = []
    
    # Check ffmpeg
    if not subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode == 0:
        missing.append('ffmpeg')
    
    # Check python3
    if not subprocess.run(['which', 'python3'], capture_output=True).returncode == 0:
        missing.append('python3')
    
    # Check if whisper is installed
    try:
        import whisper
    except ImportError:
        missing.append('openai-whisper (Python package)')
    
    # Check ollama (only if not using simple format)
    if not config.use_simple_format:
        if not subprocess.run(['which', 'ollama'], capture_output=True).returncode == 0:
            missing.append('ollama')
    
    if missing:
        print_error(f"Missing dependencies: {', '.join(missing)}")
        print_error("Run ./install.sh to install all dependencies")
        sys.exit(1)
    
    print_verbose("All dependencies found")

def check_whisper_model() -> None:
    """Validate Whisper model name"""
    valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    if config.transcription_model not in valid_models:
        print_error(f"Invalid Whisper model: {config.transcription_model}")
        print_error(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)
    
    print_verbose(f"Whisper model: {config.transcription_model} (will be downloaded if needed)")

def check_ollama_model() -> None:
    """Check if Ollama model is available and find working alternatives"""
    if config.use_simple_format:
        return
    
    # Auto-select model if user didn't specify one (using default)
    if config.format_model == "qwen2.5:7b" and not config.auto_selected_model:
        print_verbose("No specific model requested, using auto-selection...")
        auto_model = auto_select_and_pull_model()
        if auto_model:
            config.format_model = auto_model
            config.auto_selected_model = True
        else:
            print_warning("Auto-selection failed, testing available models...")
    
    # Preferred models in order of preference (smaller models first for memory efficiency)
    fallback_models = [
        config.format_model,  # User's requested model first
        'qwen2.5:7b',         # Best balance
        'phi3:mini',          # 3.8B - very efficient
        'qwen2.5:3b',         # 3B - fast and capable
        'gemma2:2b',          # 2B - small and efficient
        'llama3.2:3b',        # 3B - newer llama
        'mistral',            # 7B - good balance
        'llama3.1:8b',        # 8B - better quality
        'llama2',             # 7B - fallback
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for model in fallback_models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)
    
    config.available_format_models = []
    
    try:
        # Get list of installed models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed_models = set()
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                installed_models.add(model_name.split(':')[0])  # Remove tag
        
        print_verbose(f"Installed Ollama models: {', '.join(sorted(installed_models))}")
        
        # Test each model to see if it can actually load
        for model in unique_models:
            model_base = model.split(':')[0]
            
            # Skip if not installed
            if model_base not in installed_models:
                print_verbose(f"  Model '{model}' not installed, skipping")
                continue
            
            # Try a minimal test to see if model can load
            print_verbose(f"  Testing model '{model}'...")
            
            try:
                test_response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": model,
                        "prompt": "Hi",  # Minimal prompt
                        "stream": False,
                        "options": {
                            "num_predict": 5  # Only generate 5 tokens for testing
                        }
                    },
                    timeout=20  # Reduced timeout for faster testing
                )
                
                if test_response.status_code == 200:
                    response_data = test_response.json()
                    if 'response' in response_data:
                        config.available_format_models.append(model)
                        print_verbose(f"    ✓ Model '{model}' is working")
                        
                        # Set the first working model as current if not set
                        if config.current_format_model is None:
                            config.current_format_model = model
                            if model != config.format_model:
                                print_warning(f"Requested model '{config.format_model}' not available or won't load")
                                print_status(f"Using '{model}' instead")
                    else:
                        print_verbose(f"    ✗ Model '{model}' returned invalid response")
                else:
                    error_msg = test_response.json().get('error', 'Unknown error')
                    print_verbose(f"    ✗ Model '{model}' failed: {error_msg}")
                    
            except requests.exceptions.Timeout:
                print_verbose(f"    ✗ Model '{model}' test timeout")
            except Exception as e:
                print_verbose(f"    ✗ Model '{model}' test failed: {e}")
        
        if not config.available_format_models:
            # No models work, try to pull the requested model
            print_warning(f"No working Ollama models found")
            print_status(f"Attempting to pull model '{config.format_model}'...")
            
            pull_result = subprocess.run(['ollama', 'pull', config.format_model], 
                                        capture_output=True, text=True)
            if pull_result.returncode == 0:
                config.available_format_models = [config.format_model]
                config.current_format_model = config.format_model
                print_status(f"✓ Model '{config.format_model}' pulled successfully")
            else:
                print_error(f"Failed to pull Ollama model: {config.format_model}")
                print_error("Available models can be found at: https://ollama.ai/library")
                print_status("Will fall back to simple formatting for all videos")
        else:
            if len(config.available_format_models) > 1:
                print_verbose(f"Found {len(config.available_format_models)} working models: {', '.join(config.available_format_models)}")
            
    except Exception as e:
        print_error(f"Failed to check Ollama models: {e}")
        print_status("Will fall back to simple formatting for all videos")

################################################################################
# Video Processing Functions
################################################################################

def has_audio_track(video_file: str) -> bool:
    """Check if video file has an audio track"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return len(result.stdout.strip().split('\n')) > 0 and result.stdout.strip() != ''
    except Exception:
        return False

def extract_audio(video_file: str, audio_file: str) -> bool:
    """Extract audio from video file"""
    print_status(f"Step 1/3: Extracting audio from: {os.path.basename(video_file)}")
    
    try:
        cmd = [
            'ffmpeg', '-i', video_file,
            '-vn',  # no video
            '-acodec', 'aac',  # use AAC codec
            '-b:a', '64k',  # bitrate suitable for voice
            '-ar', '16000',  # sample rate optimized for speech
            '-ac', '1',  # mono audio
            '-y',  # overwrite output file
            audio_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status("  ✓ Audio extracted successfully")
            return True
        else:
            print_error(f"Failed to extract audio from: {video_file}")
            if config.verbose:
                print_verbose(f"  ffmpeg stderr: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Failed to extract audio: {e}")
        return False

def transcribe_audio(audio_file: str, text_file: str) -> bool:
    """Transcribe audio file using Whisper"""
    print_status(f"Step 2/3: Transcribing audio with Whisper ({config.transcription_model} model)")
    print_status("  This may take several minutes depending on audio length and model size...")
    
    try:
        import whisper
        import torch
        
        # Determine device
        device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        
        if config.use_gpu and not torch.cuda.is_available():
            print_warning("  CUDA not available, falling back to CPU")
            device = "cpu"
        
        if device == "cuda":
            print_verbose("  Using GPU acceleration")
        else:
            print_verbose("  Using CPU (slower, consider --gpu flag for faster processing)")
        
        start_time = time.time()
        
        # Get audio duration
        try:
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
            ]
            duration_output = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL)
            duration = float(duration_output.decode().strip())
            minutes = int(duration / 60)
            seconds = int(duration % 60)
            print_status(f"  Audio duration: {minutes}m {seconds}s")
        except:
            pass
        
        # Load model
        print_verbose(f"  Loading Whisper model '{config.transcription_model}' on {device}...")
        model = whisper.load_model(config.transcription_model, device=device)
        print_verbose("  Model loaded successfully")
        
        # Transcribe
        print_status("  Transcribing (this may take a while)...")
        result = model.transcribe(audio_file, verbose=config.verbose)
        
        # Write output
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(result['text'].strip())
        
        end_time = time.time()
        duration_sec = int(end_time - start_time)
        
        print_status(f"  ✓ Transcription complete (took {duration_sec}s)")
        
        # Show detected language
        if 'language' in result:
            print_verbose(f"  Detected language: {result['language']}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to transcribe audio: {e}")
        return False

def transcribe_audio_batch(audio_files: List[Tuple[str, str]]) -> int:
    """Batch transcribe multiple audio files with model loaded once
    
    Args:
        audio_files: List of (audio_file, text_file) tuples
        
    Returns:
        Number of successfully transcribed files
    """
    if not audio_files:
        return 0
    
    print_status("=" * 70)
    print_status("BATCH TRANSCRIPTION MODE - Processing all audio files")
    print_status(f"Total files to transcribe: {len(audio_files)}")
    print_status("=" * 70)
    
    try:
        import whisper
        import torch
        
        # Determine device
        device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        
        if config.use_gpu and not torch.cuda.is_available():
            print_warning("  CUDA not available, falling back to CPU")
            device = "cpu"
        
        print_status(f"Using device: {device.upper()}")
        
        # Load model ONCE for all files
        print_status(f"Loading Whisper model '{config.transcription_model}' (will be used for all {len(audio_files)} files)...")
        batch_start = time.time()
        model = whisper.load_model(config.transcription_model, device=device)
        load_time = time.time() - batch_start
        print_status(f"✓ Model loaded in {int(load_time)}s")
        print()
        
        success_count = 0
        total_transcription_time = 0
        
        # Transcribe each file
        for idx, (audio_file, text_file) in enumerate(audio_files, 1):
            basename = os.path.basename(audio_file)
            print_status(f"[{idx}/{len(audio_files)}] Transcribing: {basename}")
            
            try:
                # Get audio duration
                try:
                    duration_cmd = [
                        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
                    ]
                    duration_output = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL)
                    duration = float(duration_output.decode().strip())
                    minutes = int(duration / 60)
                    seconds = int(duration % 60)
                    print_verbose(f"  Audio duration: {minutes}m {seconds}s")
                except:
                    pass
                
                # Transcribe
                start_time = time.time()
                result = model.transcribe(audio_file, verbose=False)
                
                # Write output
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'].strip())
                
                end_time = time.time()
                duration_sec = int(end_time - start_time)
                total_transcription_time += duration_sec
                
                print_status(f"  ✓ Complete (took {duration_sec}s)")
                
                # Show detected language
                if config.verbose and 'language' in result:
                    print_verbose(f"  Detected language: {result['language']}")
                
                success_count += 1
                
            except Exception as e:
                print_error(f"  Failed: {e}")
                continue
            
            print()
        
        # Summary
        batch_end = time.time()
        total_time = int(batch_end - batch_start)
        
        print_status("=" * 70)
        print_status("BATCH TRANSCRIPTION SUMMARY")
        print_status("=" * 70)
        print_status(f"Total files processed: {success_count}/{len(audio_files)}")
        print_status(f"Model load time: {int(load_time)}s")
        print_status(f"Transcription time: {total_transcription_time}s")
        print_status(f"Total time: {total_time}s")
        
        if len(audio_files) > 1:
            saved_time = int(load_time * (len(audio_files) - 1))
            print_status(f"⚡ Time saved by batching: ~{saved_time}s ({saved_time // 60}m {saved_time % 60}s)")
        
        print_status("=" * 70)
        print()
        
        return success_count
        
    except Exception as e:
        print_error(f"Batch transcription failed: {e}")
        return 0

def format_text_simple(text_file: str, md_file: str) -> bool:
    """Simple text formatting without LLM"""
    print_status("Step 3/3: Formatting text (simple mode)")
    
    try:
        # Read transcription
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get basename
        basename = os.path.splitext(os.path.basename(text_file))[0]
        
        # Format text
        import textwrap
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        # Wrap text at 80 characters
        wrapped = textwrap.fill(text, width=80)
        
        # Create markdown
        md_content = f"""# Transcription: {basename}

---

{wrapped}
"""
        
        # Write output
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print_status("  ✓ Formatting complete")
        return True
        
    except Exception as e:
        print_error(f"Failed to format text: {e}")
        return False

def format_text_llm(text_file: str, md_file: str) -> bool:
    """Format text using Ollama LLM with automatic fallback to smaller models"""
    
    # Determine which model to try first
    models_to_try = []
    
    if config.current_format_model:
        models_to_try.append(config.current_format_model)
    
    # Add other available models as fallbacks
    if config.available_format_models:
        for model in config.available_format_models:
            if model not in models_to_try:
                models_to_try.append(model)
    
    # If no models configured, fall back immediately
    if not models_to_try:
        print_verbose("  No Ollama models available")
        return False
    
    # Read transcription once
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            transcription = f.read()
    except Exception as e:
        print_error(f"Failed to read transcription file: {e}")
        return False
    
    text_length = len(transcription)
    
    if text_length == 0:
        print_error("Transcription file is empty")
        return False
    
    # Create prompt
    prompt = f"""You are a text formatting assistant. Format the following transcription into a well-structured Markdown document. Add appropriate headers, paragraphs, and formatting. Do not add any content that is not in the original transcription. Only restructure and format the existing text.

Transcription:
{transcription}

Please provide the formatted Markdown:"""
    
    # Try each model in order
    for model_index, model in enumerate(models_to_try):
        if model_index == 0:
            print_status(f"Step 3/3: Formatting text with LLM ({model})")
        else:
            print_status(f"  Trying alternative model: {model}")
        
        print_status("  Sending to Ollama for intelligent formatting...")
        print_verbose(f"  Transcription length: {text_length} characters")
        
        try:
            # Check if Ollama is running (only on first attempt)
            if model_index == 0:
                try:
                    response = requests.get('http://localhost:11434/api/tags', timeout=5)
                    if response.status_code != 200:
                        raise Exception("Ollama not responding")
                except Exception:
                    print_error("Ollama service is not responding on localhost:11434")
                    print_error("  - Check if Ollama is running: systemctl status ollama (or: ollama serve)")
                    print_error("  - Try: ollama serve (in another terminal)")
                    print_error("  - Or use --simple-format to skip LLM formatting")
                    return False
            
            start_time = time.time()
            
            # Call Ollama API
            print_verbose(f"  Calling Ollama API with model '{model}'...")
            
            api_data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=api_data,
                    timeout=600
                )
                
                if response.status_code != 200:
                    error_msg = "Unknown error"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg = error_data['error']
                    except:
                        pass
                    
                    print_error(f"Model '{model}' returned HTTP {response.status_code}: {error_msg}")
                    
                    # Check if it's a memory error - try next model
                    if "memory" in error_msg.lower() or "load" in error_msg.lower():
                        if model_index < len(models_to_try) - 1:
                            print_verbose(f"  Memory issue with '{model}', trying smaller model...")
                            continue
                    
                    # Other errors - try next model or fail
                    if model_index < len(models_to_try) - 1:
                        continue
                    else:
                        return False
                
                # Parse response
                print_verbose("  Parsing Ollama response...")
                response_data = response.json()
                
                if 'response' not in response_data:
                    print_error("Invalid JSON response from Ollama")
                    if 'error' in response_data:
                        print_error(f"  Error from Ollama: {response_data['error']}")
                    
                    # Try next model
                    if model_index < len(models_to_try) - 1:
                        continue
                    else:
                        return False
                
                # Save formatted output
                formatted_text = response_data['response']
                
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
                
                # Verify output
                if not os.path.exists(md_file) or os.path.getsize(md_file) == 0:
                    print_error("Output file is empty or was not created")
                    
                    # Try next model
                    if model_index < len(models_to_try) - 1:
                        continue
                    else:
                        return False
                
                end_time = time.time()
                duration = int(end_time - start_time)
                
                # Update current working model if we switched
                if model != config.current_format_model:
                    print_status(f"  Model '{model}' worked successfully, will use for remaining videos")
                    config.current_format_model = model
                
                print_status(f"  ✓ Formatting complete (took {duration}s)")
                return True
                
            except requests.exceptions.Timeout:
                print_error(f"Model '{model}' request timeout (took longer than 600s)")
                if model_index < len(models_to_try) - 1:
                    continue
                else:
                    return False
                    
            except requests.exceptions.ConnectionError:
                print_error("Failed to connect to Ollama (is it running on port 11434?)")
                return False
                
            except Exception as e:
                print_error(f"Model '{model}' request failed: {e}")
                if model_index < len(models_to_try) - 1:
                    continue
                else:
                    return False
            
        except Exception as e:
            print_error(f"Failed to format with model '{model}': {e}")
            if model_index < len(models_to_try) - 1:
                continue
            else:
                return False
    
    # All models failed
    return False

def format_text(text_file: str, md_file: str) -> bool:
    """Format text using LLM or simple formatting"""
    if config.use_simple_format:
        return format_text_simple(text_file, md_file)
    
    # Try LLM formatting first
    if not format_text_llm(text_file, md_file):
        print_warning("LLM formatting failed, falling back to simple formatting")
        return format_text_simple(text_file, md_file)
    
    return True

def process_video(video_file: str) -> bool:
    """Process a single video file"""
    print()
    print_status("━" * 70)
    print_status(f"Processing: {os.path.basename(video_file)}")
    print_status(f"Location: {os.path.dirname(video_file)}")
    
    # Determine output file paths
    dir_path = os.path.dirname(video_file)
    filename = os.path.basename(video_file)
    basename = os.path.splitext(filename)[0]
    
    audio_file = os.path.join(dir_path, f"{basename}.aac")
    text_file = os.path.join(dir_path, f"{basename}.txt")
    md_file = os.path.join(dir_path, f"{basename}.md")
    
    # Check if we should skip processing entirely
    if config.skip_transcoding:
        # If both .txt and .md exist, skip everything
        if os.path.exists(text_file) and os.path.exists(md_file):
            print_status(f"✓ Transcription and formatted output already exist: {basename}.txt, {basename}.md")
            print_status("  Skipping all processing (--skip-transcoding)")
            print_status("━" * 70)
            print_status(f"✓✓✓ SKIPPED: {os.path.basename(video_file)} (already processed)")
            stats.skipped_videos += 1
            return True
        
        # If only .txt exists, skip transcoding but do formatting
        if os.path.exists(text_file):
            print_status(f"✓ Transcription file exists: {basename}.txt")
            print_status("  Skipping audio extraction and transcription (--skip-transcoding)")
            
            # Only do formatting
            if not format_text(text_file, md_file):
                stats.failed_videos += 1
                return False
            
            print_status("━" * 70)
            print_status(f"✓✓✓ COMPLETED: {os.path.basename(video_file)} (formatting only)")
            print_status(f"    Generated: {basename}.md")
            stats.processed_videos += 1
            return True
    
    # Check if video has audio track
    if not has_audio_track(video_file):
        print_warning("No audio track found, skipping")
        stats.skipped_videos += 1
        return False
    
    print_status("✓ Audio track detected")
    
    # Extract audio
    if not extract_audio(video_file, audio_file):
        stats.failed_videos += 1
        return False
    
    # Transcribe audio
    if not transcribe_audio(audio_file, text_file):
        stats.failed_videos += 1
        return False
    
    # Format text
    if not format_text(text_file, md_file):
        stats.failed_videos += 1
        return False
    
    print_status("━" * 70)
    print_status(f"✓✓✓ COMPLETED: {os.path.basename(video_file)}")
    print_status(f"    Generated: {basename}.aac, {basename}.txt, {basename}.md")
    stats.processed_videos += 1
    return True

################################################################################
# Main Functions
################################################################################

def find_videos(root: str) -> List[str]:
    """Find all video files in directory tree"""
    print_verbose(f"Searching for videos in: {root}")
    
    videos = []
    root_path = Path(root)
    
    for ext in VIDEO_EXTENSIONS:
        # Case-insensitive search
        for pattern in [f"*.{ext}", f"*.{ext.upper()}"]:
            videos.extend(root_path.rglob(pattern))
    
    # Filter out macOS resource fork files
    videos = [str(v) for v in videos if not os.path.basename(str(v)).startswith('._')]
    
    return sorted(videos)

def process_all_videos(root: str) -> None:
    """Process all video files in directory"""
    videos = find_videos(root)
    
    stats.total_videos = len(videos)
    
    if stats.total_videos == 0:
        print_warning(f"No video files found in: {root}")
        return
    
    print_status(f"Found {stats.total_videos} video file(s)")
    print()
    
    # Batch mode: Process in two phases
    if config.batch_mode and stats.total_videos > 1:
        process_all_videos_batch(videos)
    else:
        # Sequential mode: Process each video completely
        for i, video in enumerate(videos, 1):
            print()
            print_status("=" * 70)
            print_status(f"Video {i} of {stats.total_videos}")
            print_status("=" * 70)
            process_video(video)

def process_all_videos_batch(videos: List[str]) -> None:
    """Process all videos in batch mode - two phases"""
    print_status("=" * 70)
    print_status("BATCH MODE ENABLED - Processing in 2 phases for optimal GPU usage")
    print_status("=" * 70)
    print()
    
    # Phase 1: Extract audio and transcribe all videos
    print_status("╔" + "═" * 68 + "╗")
    print_status("║" + " " * 20 + "PHASE 1: AUDIO EXTRACTION & TRANSCRIPTION" + " " * 7 + "║")
    print_status("╚" + "═" * 68 + "╝")
    print()
    
    audio_files_to_transcribe = []
    videos_to_format = []
    
    for i, video_file in enumerate(videos, 1):
        print_status(f"[{i}/{len(videos)}] Preparing: {os.path.basename(video_file)}")
        
        # Determine output file paths
        dir_path = os.path.dirname(video_file)
        filename = os.path.basename(video_file)
        basename = os.path.splitext(filename)[0]
        
        audio_file = os.path.join(dir_path, f"{basename}.aac")
        text_file = os.path.join(dir_path, f"{basename}.txt")
        md_file = os.path.join(dir_path, f"{basename}.md")
        
        # Check skip conditions
        if config.skip_transcoding:
            if os.path.exists(text_file) and os.path.exists(md_file):
                print_status(f"  ✓ Already processed, skipping")
                stats.skipped_videos += 1
                continue
            
            if os.path.exists(text_file):
                print_status(f"  ✓ Transcription exists, will format only")
                videos_to_format.append((video_file, text_file, md_file))
                continue
        
        # Check if video has audio
        if not has_audio_track(video_file):
            print_warning(f"  No audio track, skipping")
            stats.skipped_videos += 1
            continue
        
        # Extract audio
        print_status(f"  Extracting audio...")
        if not extract_audio(video_file, audio_file):
            print_error(f"  Failed to extract audio")
            stats.failed_videos += 1
            continue
        
        print_status(f"  ✓ Audio extracted: {basename}.aac")
        audio_files_to_transcribe.append((audio_file, text_file))
        videos_to_format.append((video_file, text_file, md_file))
    
    print()
    
    # Batch transcribe all audio files
    if audio_files_to_transcribe:
        success_count = transcribe_audio_batch(audio_files_to_transcribe)
        
        if success_count < len(audio_files_to_transcribe):
            failed = len(audio_files_to_transcribe) - success_count
            stats.failed_videos += failed
            print_warning(f"{failed} transcription(s) failed")
    
    # Phase 2: Format all text files
    print_status("╔" + "═" * 68 + "╗")
    print_status("║" + " " * 24 + "PHASE 2: TEXT FORMATTING" + " " * 20 + "║")
    print_status("╚" + "═" * 68 + "╝")
    print()
    
    if not videos_to_format:
        print_warning("No files to format")
        return
    
    print_status(f"Formatting {len(videos_to_format)} transcription(s)...")
    print()
    
    for i, (video_file, text_file, md_file) in enumerate(videos_to_format, 1):
        basename = os.path.splitext(os.path.basename(video_file))[0]
        print_status(f"[{i}/{len(videos_to_format)}] Formatting: {basename}")
        
        # Check if text file exists
        if not os.path.exists(text_file):
            print_error(f"  Text file not found: {basename}.txt")
            stats.failed_videos += 1
            continue
        
        # Format the text
        if format_text(text_file, md_file):
            print_status(f"  ✓ Generated: {basename}.md")
            stats.processed_videos += 1
        else:
            print_error(f"  Failed to format")
            stats.failed_videos += 1
        
        print()
    
    print_status("=" * 70)
    print_status("BATCH PROCESSING COMPLETE")
    print_status("=" * 70)

def show_summary() -> None:
    """Display processing summary"""
    print()
    print("=" * 70)
    print_status("Processing Summary")
    print("=" * 70)
    print(f"Total videos found:     {stats.total_videos}")
    print(f"Successfully processed: {stats.processed_videos}")
    print(f"Skipped (no audio):     {stats.skipped_videos}")
    print(f"Failed:                 {stats.failed_videos}")
    print("=" * 70)

################################################################################
# Argument Parsing
################################################################################

def parse_arguments() -> None:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Video/Audio Transcoding Script',
        add_help=False,  # We'll handle help ourselves
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-g', '--gpu', action='store_true',
                       help='Force GPU acceleration')
    parser.add_argument('-c', '--cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect CUDA)')
    parser.add_argument('-t', '--transcription-model', default='base',
                       help='Whisper model to use (default: base)')
    parser.add_argument('-f', '--format-model', default=None,
                       help='Ollama model for text formatting (default: auto-select based on GPU)')
    parser.add_argument('-s', '--simple-format', action='store_true',
                       help='Use simple formatting instead of LLM')
    parser.add_argument('--skip-transcoding', action='store_true',
                       help='Skip processing if output files exist (.txt and/or .md)')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Batch process all videos (load Whisper once, 5-25%% faster for large batches)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help message')
    parser.add_argument('directory', nargs='?',
                       help='Root directory to search for video files')
    
    args = parser.parse_args()
    
    # Handle help
    if args.help:
        print(show_help())
        sys.exit(0)
    
    # Check if directory is provided
    if not args.directory:
        print_error("No directory specified")
        print(show_help())
        sys.exit(1)
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        print_error(f"Directory does not exist: {args.directory}")
        sys.exit(1)
    
    # Auto-detect CUDA and enable GPU by default if available (unless --cpu specified)
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except:
        pass
    
    # Update global config
    if args.cpu:
        config.use_gpu = False
    elif args.gpu:
        config.use_gpu = True
    else:
        # Auto-enable GPU if CUDA is available
        config.use_gpu = cuda_available
        if cuda_available:
            print_verbose("Auto-detected CUDA, enabling GPU acceleration")
    
    config.transcription_model = args.transcription_model
    
    # Handle format model
    if args.format_model:
        config.format_model = args.format_model
        config.auto_selected_model = False  # User explicitly specified
    else:
        config.format_model = "qwen2.5:7b"  # Default, will trigger auto-selection
        config.auto_selected_model = False
    
    config.use_simple_format = args.simple_format
    config.skip_transcoding = args.skip_transcoding
    config.batch_mode = args.batch_mode
    config.verbose = args.verbose
    config.root_dir = os.path.abspath(args.directory)

################################################################################
# Main Entry Point
################################################################################

def main() -> None:
    """Main entry point"""
    print("=" * 70)
    print("Video/Audio Transcoding Script")
    print("=" * 70)
    print()
    
    parse_arguments()
    
    print_verbose("Configuration:")
    print_verbose(f"  Root directory: {config.root_dir}")
    print_verbose(f"  GPU acceleration: {config.use_gpu}")
    print_verbose(f"  Transcription model: {config.transcription_model}")
    print_verbose(f"  Format model: {config.format_model}")
    print_verbose(f"  Simple formatting: {config.use_simple_format}")
    print_verbose(f"  Skip transcoding: {config.skip_transcoding}")
    print()
    
    check_dependencies()
    check_whisper_model()
    check_ollama_model()
    
    print()
    process_all_videos(config.root_dir)
    
    show_summary()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
