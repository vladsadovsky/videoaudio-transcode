#!/bin/bash

################################################################################
# Video/Audio Transcoding Installation Script
# Installs all required dependencies on Ubuntu/Debian systems
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu/Debian
check_system() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
            print_warning "This script is designed for Ubuntu/Debian. Your system: $ID"
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Update package list
update_packages() {
    print_status "Updating package list..."
    sudo apt update
}

# Install FFmpeg
install_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        print_status "FFmpeg is already installed: $(ffmpeg -version | head -n 1)"
    else
        print_status "Installing FFmpeg..."
        sudo apt install -y ffmpeg
        print_status "FFmpeg installed successfully"
    fi
}

# Install jq for JSON processing
install_jq() {
    if command -v jq &> /dev/null; then
        print_status "jq is already installed: $(jq --version)"
    else
        print_status "Installing jq..."
        sudo apt install -y jq
        print_status "jq installed successfully"
    fi
}

# Install basic utilities (usually pre-installed)
install_basic_utils() {
    print_status "Ensuring basic utilities are installed..."
    sudo apt install -y sed gawk coreutils findutils
}

# Install Python3 and pip
install_python() {
    if command -v python3 &> /dev/null; then
        print_status "Python3 is already installed: $(python3 --version)"
    else
        print_status "Installing Python3..."
        sudo apt install -y python3 python3-pip python3-venv
        print_status "Python3 installed successfully"
    fi
    
    # Ensure pip and venv are installed
    if ! python3 -m pip --version &> /dev/null 2>&1; then
        print_status "Installing pip..."
        sudo apt install -y python3-pip
    fi
    
    if ! python3 -m venv --help &> /dev/null 2>&1; then
        print_status "Installing python3-venv..."
        sudo apt install -y python3-venv python3-full
    fi
}

# Install OpenAI Whisper (Python package)
install_whisper() {
    print_status "Installing OpenAI Whisper Python package..."
    
    # Install system dependencies for audio processing
    sudo apt install -y python3-dev
    
    # Check if we're in an externally-managed environment
    if python3 -c "import sys; sys.exit(0 if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else 1)" 2>/dev/null; then
        # Already in a virtual environment
        print_verbose "Using existing virtual environment"
        python3 -m pip install --upgrade openai-whisper
    elif command -v pipx &> /dev/null; then
        # Use pipx if available
        print_status "Using pipx for isolated installation..."
        pipx install openai-whisper
    else
        # Try installing pipx first
        print_status "Installing pipx for isolated package management..."
        sudo apt install -y pipx
        pipx ensurepath
        
        # Install whisper via pipx
        print_status "Installing whisper via pipx..."
        pipx install openai-whisper
        
        print_warning "Please run: source ~/.bashrc or restart your terminal for pipx PATH updates"
    fi
    
    print_status "OpenAI Whisper installed successfully"
    print_status "Models will be downloaded automatically on first use"
    print_status "Available models: tiny, base, small, medium, large-v3"
}

# Install Ollama via SNAP
install_ollama() {
    if command -v ollama &> /dev/null; then
        print_status "Ollama is already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    else
        print_status "Installing Ollama via SNAP..."
        
        # Check if snap is installed
        if ! command -v snap &> /dev/null; then
            print_status "Installing snapd..."
            sudo apt install -y snapd
        fi
        
        sudo snap install ollama
        print_status "Ollama installed successfully"
    fi
    
    print_status "Pulling default Ollama model (llama2)..."
    print_warning "This may take a while depending on your internet connection..."
    
    # Start ollama service if not running
    if ! pgrep -x "ollama" > /dev/null; then
        ollama serve &
        sleep 5
    fi
    
    ollama pull llama2 || print_warning "Could not pull llama2 model. You can pull it manually later with: ollama pull llama2"
}

# Install LM Studio CLI (optional)
install_lmstudio() {
    print_status "LM Studio CLI installation..."
    print_warning "LM Studio CLI must be installed manually from https://lmstudio.ai/"
    print_warning "This is optional and not required for basic functionality"
    
    read -p "Open LM Studio website in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open "https://lmstudio.ai/" &
        else
            print_warning "Please visit https://lmstudio.ai/ to download LM Studio"
        fi
    fi
}

# Main installation function
main() {
    print_status "Starting installation of video/audio transcoding dependencies..."
    echo
    
    check_system
    update_packages
    
    install_ffmpeg
    install_jq
    install_basic_utils
    install_python
    install_whisper
    install_ollama
    install_lmstudio
    
    echo
    print_status "Installation complete!"
    echo
    print_status "Next steps:"
    echo "  1. Run: ./transcode.sh --help"
    echo "  2. Whisper models will be downloaded automatically on first use"
    echo "  3. To pull additional Ollama models:"
    echo "     ollama pull mistral"
    echo
}

# Run main function
main
