#!/bin/bash

################################################################################
# Installation Script for Video/Audio Transcoding (Python Version)
# Installs all required dependencies
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

check_os() {
    print_header "Checking Operating System"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Detected Linux"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Detected macOS"
        OS="macos"
    else
        print_warning "Unknown OS: $OSTYPE"
        print_warning "Installation may not work correctly"
        OS="unknown"
    fi
}

install_system_packages() {
    print_header "Installing System Packages"
    
    if [[ "$OS" == "linux" ]]; then
        # Detect package manager
        if command -v apt-get &> /dev/null; then
            print_status "Using apt package manager"
            sudo apt-get update
            sudo apt-get install -y \
                ffmpeg \
                python3 \
                python3-pip \
                python3-venv
            
        elif command -v dnf &> /dev/null; then
            print_status "Using dnf package manager"
            sudo dnf install -y \
                ffmpeg \
                python3 \
                python3-pip
            
        elif command -v yum &> /dev/null; then
            print_status "Using yum package manager"
            sudo yum install -y \
                ffmpeg \
                python3 \
                python3-pip
            
        elif command -v pacman &> /dev/null; then
            print_status "Using pacman package manager"
            sudo pacman -S --noconfirm \
                ffmpeg \
                python \
                python-pip
        else
            print_error "No supported package manager found"
            print_error "Please install ffmpeg and python3 manually"
            exit 1
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Please install from https://brew.sh"
            exit 1
        fi
        
        print_status "Using Homebrew"
        brew install ffmpeg python3
    fi
    
    print_status "✓ System packages installed"
}

install_python_packages() {
    print_header "Installing Python Packages"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    # Install required Python packages
    print_status "Installing OpenAI Whisper..."
    python3 -m pip install --upgrade openai-whisper
    
    print_status "Installing requests..."
    python3 -m pip install --upgrade requests
    
    # Install PyTorch with appropriate version
    print_status "Checking PyTorch installation..."
    
    if python3 -c "import torch" 2>/dev/null; then
        print_status "✓ PyTorch already installed"
    else
        print_status "Installing PyTorch..."
        
        # Check for CUDA availability
        if command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
            python3 -m pip install torch torchvision torchaudio
        else
            print_status "No NVIDIA GPU detected, installing CPU-only PyTorch..."
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    
    print_status "✓ Python packages installed"
}

install_ollama() {
    print_header "Installing Ollama"
    
    if command -v ollama &> /dev/null; then
        print_status "✓ Ollama already installed"
        ollama --version
    else
        print_status "Installing Ollama..."
        
        if [[ "$OS" == "linux" ]]; then
            curl -fsSL https://ollama.com/install.sh | sh
        elif [[ "$OS" == "macos" ]]; then
            print_status "Downloading Ollama for macOS..."
            curl -L https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama.zip
            unzip -o /tmp/Ollama.zip -d /Applications/
            rm /tmp/Ollama.zip
            print_status "Please start Ollama from /Applications/Ollama.app"
        fi
        
        print_status "✓ Ollama installed"
    fi
    
    # Try to start Ollama service (Linux only)
    if [[ "$OS" == "linux" ]]; then
        if systemctl is-active --quiet ollama; then
            print_status "✓ Ollama service is running"
        else
            print_status "Starting Ollama service..."
            if sudo systemctl start ollama 2>/dev/null; then
                sudo systemctl enable ollama
                print_status "✓ Ollama service started and enabled"
            else
                print_warning "Could not start Ollama service automatically"
                print_status "You can start it manually with: ollama serve"
            fi
        fi
    fi
    
    # Pull optimal models based on GPU VRAM
    print_status "Detecting GPU and downloading optimal Ollama models..."
    
    local models_to_download=()
    local primary_model=""
    
    # Try to detect GPU and select models
    if command -v nvidia-smi &> /dev/null; then
        gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_status "Detected GPU with ${gpu_mem}MB VRAM ($(echo "scale=1; $gpu_mem/1024" | bc)GB)"
        
        # Select models based on VRAM with fallback options
        # Note: Reserve ~2GB for Whisper during transcription
        
        if [ "$gpu_mem" -ge 24000 ]; then
            # 24GB+ VRAM - Download multiple high-quality models
            print_status "Large GPU detected - downloading multiple high-quality models..."
            primary_model="qwen2.5:14b"
            models_to_download=("qwen2.5:14b" "llama3.1:13b" "qwen2.5:7b" "mistral")
            
        elif [ "$gpu_mem" -ge 16000 ]; then
            # 16-24GB VRAM - Download 14B model + fallbacks
            print_status "High-end GPU detected - downloading 14B model with fallbacks..."
            primary_model="qwen2.5:14b"
            models_to_download=("qwen2.5:14b" "qwen2.5:7b" "mistral")
            
        elif [ "$gpu_mem" -ge 12000 ]; then
            # 12-16GB VRAM - Download 14B model (will work) + 7B fallback
            print_status "Mid-high GPU detected (${gpu_mem}MB) - downloading 14B model..."
            primary_model="qwen2.5:14b"
            models_to_download=("qwen2.5:14b" "qwen2.5:7b")
            
        elif [ "$gpu_mem" -ge 8000 ]; then
            # 8-12GB VRAM - Download 7B models
            print_status "Mid-range GPU detected - downloading 7B models..."
            primary_model="qwen2.5:7b"
            models_to_download=("qwen2.5:7b" "mistral" "phi3:mini")
            
        elif [ "$gpu_mem" -ge 6000 ]; then
            # 6-8GB VRAM - Download 7B + smaller fallback
            print_status "Moderate GPU detected - downloading 7B model with fallback..."
            primary_model="qwen2.5:7b"
            models_to_download=("qwen2.5:7b" "phi3:mini")
            
        elif [ "$gpu_mem" -ge 4000 ]; then
            # 4-6GB VRAM - Download smaller efficient models
            print_status "Modest GPU detected - downloading efficient models..."
            primary_model="phi3:mini"
            models_to_download=("phi3:mini" "gemma2:2b")
            
        else
            # <4GB VRAM - Download minimal models
            print_status "Limited GPU detected - downloading compact models..."
            primary_model="gemma2:2b"
            models_to_download=("gemma2:2b")
        fi
        
        print_status "Will download: ${models_to_download[*]}"
        print_status "Primary model: $primary_model"
        echo ""
        
    else
        # No GPU detected - download CPU-friendly models
        print_status "No NVIDIA GPU detected - downloading CPU-friendly models..."
        primary_model="qwen2.5:7b"
        models_to_download=("qwen2.5:7b" "phi3:mini")
    fi
    
    # Download each model
    local download_count=0
    local failed_count=0
    
    for model in "${models_to_download[@]}"; do
        print_status "Downloading model: $model"
        
        # Check if already exists
        if ollama list | grep -q "^$model"; then
            print_status "  ✓ Model '$model' already downloaded"
            ((download_count++))
        else
            print_status "  Pulling $model (this may take several minutes)..."
            if timeout 600 ollama pull "$model" 2>&1 | grep -E "(pulling|success|✓)" || true; then
                if ollama list | grep -q "^$model"; then
                    print_status "  ✓ Model '$model' downloaded successfully"
                    ((download_count++))
                else
                    print_warning "  ✗ Model '$model' pull completed but not found in list"
                    ((failed_count++))
                fi
            else
                print_warning "  ✗ Failed to pull model '$model' (timeout or error)"
                ((failed_count++))
            fi
        fi
        echo ""
    done
    
    # Summary
    echo ""
    print_status "═══════════════════════════════════════════════════════════════════"
    print_status "Model Download Summary:"
    print_status "  Successfully downloaded/verified: $download_count model(s)"
    if [ $failed_count -gt 0 ]; then
        print_warning "  Failed downloads: $failed_count model(s)"
        print_warning "  You can manually download failed models with: ollama pull <model-name>"
    fi
    print_status "  Primary model for your system: $primary_model"
    print_status "═══════════════════════════════════════════════════════════════════"
    echo ""
    
    # Verify at least one model is available
    if [ $download_count -eq 0 ]; then
        print_error "No models were successfully downloaded!"
        print_error "Please check your internet connection and Ollama installation"
        print_warning "You can manually download models later with: ollama pull <model-name>"
    fi
}

make_executable() {
    print_header "Making Scripts Executable"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ -f "$SCRIPT_DIR/transcode.py" ]; then
        chmod +x "$SCRIPT_DIR/transcode.py"
        print_status "✓ Made transcode.py executable"
    fi
}

verify_installation() {
    print_header "Verifying Installation"
    
    local all_ok=true
    
    # Check ffmpeg
    if command -v ffmpeg &> /dev/null; then
        print_status "✓ ffmpeg: $(ffmpeg -version | head -n1)"
    else
        print_error "✗ ffmpeg not found"
        all_ok=false
    fi
    
    # Check python3
    if command -v python3 &> /dev/null; then
        print_status "✓ python3: $(python3 --version)"
    else
        print_error "✗ python3 not found"
        all_ok=false
    fi
    
    # Check whisper
    if python3 -c "import whisper" 2>/dev/null; then
        print_status "✓ openai-whisper installed"
    else
        print_error "✗ openai-whisper not installed"
        all_ok=false
    fi
    
    # Check torch
    if python3 -c "import torch" 2>/dev/null; then
        print_status "✓ PyTorch installed"
        
        # Check CUDA
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_status "  ✓ CUDA support available"
        else
            print_status "  ℹ CUDA support not available (CPU-only mode)"
        fi
    else
        print_error "✗ PyTorch not installed"
        all_ok=false
    fi
    
    # Check requests
    if python3 -c "import requests" 2>/dev/null; then
        print_status "✓ requests installed"
    else
        print_error "✗ requests not installed"
        all_ok=false
    fi
    
    # Check ollama
    if command -v ollama &> /dev/null; then
        print_status "✓ ollama: $(ollama --version 2>&1 | head -n1)"
    else
        print_warning "✗ ollama not found (optional for simple formatting)"
    fi
    
    echo ""
    if [ "$all_ok" = true ]; then
        print_status "═══════════════════════════════════════════════════════════════════"
        print_status "✓ All required dependencies installed successfully!"
        print_status "═══════════════════════════════════════════════════════════════════"
        return 0
    else
        print_error "═══════════════════════════════════════════════════════════════════"
        print_error "✗ Some dependencies are missing. Please check the errors above."
        print_error "═══════════════════════════════════════════════════════════════════"
        return 1
    fi
}

show_usage() {
    print_header "Usage Instructions"
    
    cat << 'EOF'
The installation is complete! Here's how to use the script:

BASIC USAGE:
    ./transcode.py /path/to/videos
    
    The script automatically:
    - Detects GPU and enables CUDA if available
    - Selects optimal Ollama model for your VRAM
    - No manual configuration needed!

EXAMPLES:
    # Process videos (auto-detect everything)
    ./transcode.py .

    # Force CPU only
    ./transcode.py --cpu /path/to/videos

    # Use specific Whisper model
    ./transcode.py -t large-v3 /path/to/videos
    
    # Override model selection
    ./transcode.py -f qwen2.5:14b /path/to/videos

    # Simple formatting (no LLM)
    ./transcode.py --simple-format /path/to/videos

    # Skip already processed files
    ./transcode.py --skip-transcoding /path/to/videos

    # Verbose mode (see model selection details)
    ./transcode.py -v /path/to/videos

    # Get help
    ./transcode.py --help

DOWNLOADED OLLAMA MODELS:
    The installer downloaded optimal models for your GPU.
    To see available models: ollama list
    To download additional models: ollama pull <model-name>
    
    Recommended models by VRAM:
    - 24GB+:  qwen2.5:14b, llama3.1:13b
    - 16GB:   qwen2.5:14b, qwen2.5:7b
    - 12GB:   qwen2.5:14b (tight), qwen2.5:7b (safe)
    - 8GB:    qwen2.5:7b, mistral
    - 6GB:    qwen2.5:7b, phi3:mini
    - 4GB:    phi3:mini, gemma2:2b

NOTES:
    - First transcription downloads Whisper model (~140MB for 'base')
    - GPU acceleration requires NVIDIA GPU with CUDA
    - Ollama must be running for LLM formatting
    - Start Ollama: ollama serve (Linux) or from Applications (macOS)
    - Models auto-downloaded during installation
    - Script auto-selects best model on each run

TROUBLESHOOTING:
    If model fails to load (memory error):
    - Script automatically tries smaller models
    - Or use --cpu flag to process without GPU
    - Or use -f <smaller-model> to specify manually
    
    Check GPU memory:
    nvidia-smi
    
    List downloaded models:
    ollama list
    
    Download specific model:
    ollama pull qwen2.5:7b

For more information, see README_PYTHON.md
EOF
}

main() {
    echo "======================================================================"
    echo "Video/Audio Transcoding - Installation Script (Python Version)"
    echo "======================================================================"
    
    check_os
    install_system_packages
    install_python_packages
    install_ollama
    make_executable
    
    if verify_installation; then
        show_usage
        exit 0
    else
        exit 1
    fi
}

main "$@"
