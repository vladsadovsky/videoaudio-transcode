#!/bin/bash

################################################################################
# Video/Audio Transcoding Script
# Extracts audio from videos, transcribes, and formats the text
################################################################################

# Note: Not using 'set -e' because we handle errors explicitly in each function
# and want to continue processing other videos even if one fails

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
USE_GPU=false
TRANSCRIPTION_MODEL="base"
FORMAT_MODEL="llama2"
USE_SIMPLE_FORMAT=false
SKIP_TRANSCODING=false
VERBOSE=false
ROOT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_SCRIPT="$SCRIPT_DIR/whisper_transcribe.py"

# Supported video extensions
VIDEO_EXTENSIONS=("mp4" "mkv" "avi" "mov")

# Statistics
TOTAL_VIDEOS=0
PROCESSED_VIDEOS=0
SKIPPED_VIDEOS=0
FAILED_VIDEOS=0

################################################################################
# Helper Functions
################################################################################

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

show_help() {
    cat << EOF
Video/Audio Transcoding Script

Usage: $0 [OPTIONS] <directory>

OPTIONS:
    -g, --gpu                   Use GPU acceleration (default: CPU)
    -c, --cpu                   Force CPU usage (default)
    -t, --transcription-model   Whisper model to use (default: base)
                                Options: tiny, base, small, medium, large-v3
    -f, --format-model          Ollama model for text formatting (default: llama2)
    -s, --simple-format         Use simple bash-based formatting instead of LLM
    --skip-transcoding          Skip transcription if .txt file exists, only format
    -v, --verbose               Enable verbose output
    -h, --help                  Display this help message

ARGUMENTS:
    directory                   Root directory to search for video files
                                Use '.' for current directory

EXAMPLES:
    # Process videos in current directory
    $0 .

    # Process with GPU acceleration
    $0 --gpu /path/to/videos

    # Use specific models
    $0 -t large-v3 -f mistral /path/to/videos

    # Use simple formatting (no LLM)
    $0 --simple-format /path/to/videos

    # Only reformat existing transcriptions
    $0 --skip-transcoding -f mistral /path/to/videos

SUPPORTED VIDEO FORMATS:
    .mp4, .mkv, .avi, .mov

WORKFLOW:
    1. Find all video files in directory tree
    2. Extract audio track to AAC format (unless --skip-transcoding)
    3. Transcribe audio using Whisper (unless --skip-transcoding and .txt exists)
    4. Format transcription to Markdown

OUTPUT FILES (for video.mp4):
    video.aac  - Extracted audio
    video.txt  - Raw transcription
    video.md   - Formatted Markdown

EOF
}

################################################################################
# Validation Functions
################################################################################

check_dependencies() {
    local missing=()
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        missing+=("ffmpeg")
    fi
    
    # Check python3
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi
    
    # Check if whisper is installed (Python package via pipx or pip)
    if ! python3 -c "import whisper" 2>/dev/null && ! command -v whisper &> /dev/null; then
        missing+=("openai-whisper (Python package)")
    fi
    
    # Check ollama (only if not using simple format)
    if [ "$USE_SIMPLE_FORMAT" = false ] && ! command -v ollama &> /dev/null; then
        missing+=("ollama")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi
    
    # Check if whisper script exists
    if [ ! -f "$WHISPER_SCRIPT" ]; then
        missing+=("whisper_transcribe.py")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_error "Run ./install.sh to install all dependencies"
        exit 1
    fi
    
    print_verbose "All dependencies found"
}

check_whisper_model() {
    # With Python whisper, models are downloaded automatically on first use
    # Just verify that the model name is valid
    local valid_models=("tiny" "base" "small" "medium" "large" "large-v2" "large-v3")
    local is_valid=false
    
    for model in "${valid_models[@]}"; do
        if [ "$TRANSCRIPTION_MODEL" = "$model" ]; then
            is_valid=true
            break
        fi
    done
    
    if [ "$is_valid" = false ]; then
        print_error "Invalid Whisper model: $TRANSCRIPTION_MODEL"
        print_error "Valid models: ${valid_models[*]}"
        exit 1
    fi
    
    print_verbose "Whisper model: $TRANSCRIPTION_MODEL (will be downloaded if needed)"
}

check_ollama_model() {
    if [ "$USE_SIMPLE_FORMAT" = true ]; then
        return 0
    fi
    
    if ! ollama list | grep -q "$FORMAT_MODEL"; then
        print_warning "Ollama model '$FORMAT_MODEL' not found locally"
        print_status "Attempting to pull model..."
        
        if ! ollama pull "$FORMAT_MODEL"; then
            print_error "Failed to pull Ollama model: $FORMAT_MODEL"
            print_error "Available models can be found at: https://ollama.ai/library"
            exit 1
        fi
    fi
    
    print_verbose "Ollama model found: $FORMAT_MODEL"
}

################################################################################
# Video Processing Functions
################################################################################

has_audio_track() {
    local video_file="$1"
    
    # Check if video has audio stream
    local audio_streams
    audio_streams=$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "$video_file" 2>/dev/null | wc -l)
    
    if [ "$audio_streams" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

extract_audio() {
    local video_file="$1"
    local audio_file="$2"
    
    print_status "Step 1/3: Extracting audio from: $(basename "$video_file")"
    
    # Extract audio and convert to AAC with optimized settings for voice
    # -vn: no video
    # -acodec aac: use AAC codec
    # -b:a 64k: bitrate suitable for voice
    # -ar 16000: sample rate optimized for speech
    # -ac 1: mono audio (sufficient for voice)
    
    if ffmpeg -i "$video_file" -vn -acodec aac -b:a 64k -ar 16000 -ac 1 -y "$audio_file" 2>&1 | grep -i "time=" | tail -1 | tr '\r' '\n' | tail -1 | sed 's/^/  /'; then
        print_status "  ✓ Audio extracted successfully"
        return 0
    else
        print_error "Failed to extract audio from: $video_file"
        return 1
    fi
}

transcribe_audio() {
    local audio_file="$1"
    local text_file="$2"
    
    print_status "Step 2/3: Transcribing audio with Whisper ($TRANSCRIPTION_MODEL model)"
    print_status "  This may take several minutes depending on audio length and model size..."
    
    # Determine device
    local device="cpu"
    if [ "$USE_GPU" = true ]; then
        device="cuda"
        print_verbose "  Using GPU acceleration"
    else
        print_verbose "  Using CPU (slower, consider --gpu flag for faster processing)"
    fi
    
    # Show a spinner while transcribing
    local start_time
    local end_time
    local duration
    start_time=$(date +%s)
    
    # Determine which Python to use
    local python_cmd="python3"
    
    # Check if whisper is installed via pipx and use that Python
    if [ -d "$HOME/.local/share/pipx/venvs/openai-whisper" ]; then
        python_cmd="$HOME/.local/share/pipx/venvs/openai-whisper/bin/python"
        print_verbose "  Using pipx-installed whisper environment"
    elif command -v pipx &> /dev/null && pipx list | grep -q "openai-whisper"; then
        # Find the pipx venv for whisper
        local pipx_venv
        pipx_venv=$(pipx environment --value PIPX_LOCAL_VENVS 2>/dev/null || echo "$HOME/.local/share/pipx/venvs")
        if [ -f "$pipx_venv/openai-whisper/bin/python" ]; then
            python_cmd="$pipx_venv/openai-whisper/bin/python"
            print_verbose "  Using pipx-installed whisper environment"
        fi
    fi
    
    # Call Python whisper script
    if "$python_cmd" "$WHISPER_SCRIPT" \
        --model "$TRANSCRIPTION_MODEL" \
        --device "$device" \
        "$audio_file" \
        "$text_file" 2>&1 | while IFS= read -r line; do
            echo "  $line"
        done; then
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [ -f "$text_file" ]; then
            print_status "  ✓ Transcription complete (took ${duration}s)"
            return 0
        else
            print_error "Transcription failed: output file not created"
            return 1
        fi
    else
        print_error "Failed to transcribe audio: $audio_file"
        return 1
    fi
}

format_text_simple() {
    local text_file="$1"
    local md_file="$2"
    
    print_status "Step 3/3: Formatting text (simple mode)"
    
    # Simple formatting using bash utilities
    # 1. Add title based on filename
    # 2. Break into paragraphs (double newlines for empty lines)
    # 3. Wrap text at 80 characters
    # 4. Clean up excessive whitespace
    
    local basename
    basename=$(basename "${text_file%.txt}")
    
    {
        echo "# Transcription: $basename"
        echo ""
        echo "---"
        echo ""
        
        # Process the text:
        # - Remove excessive spaces
        # - Format into paragraphs
        # - Wrap at 80 characters
        cat "$text_file" | \
            sed 's/  */ /g' | \
            sed 's/^ *//g' | \
            fmt -w 80 | \
            awk 'BEGIN{blank=0} /^$/{blank++; if(blank==1) print; next} {blank=0; print}'
    } > "$md_file"
    
    print_status "  ✓ Formatting complete"
    return 0
}

format_text_llm() {
    local text_file="$1"
    local md_file="$2"
    
    print_status "Step 3/3: Formatting text with LLM ($FORMAT_MODEL)"
    print_status "  Sending to Ollama for intelligent formatting..."
    
    # Check if Ollama is running
    if ! curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_error "Ollama service is not responding on localhost:11434"
        print_error "  - Check if Ollama is running: systemctl status ollama (or: ollama serve)"
        print_error "  - Try: ollama serve (in another terminal)"
        print_error "  - Or use --simple-format to skip LLM formatting"
        return 1
    fi
    
    # Verify model is available
    if ! curl -s http://localhost:11434/api/tags | jq -e ".models[] | select(.name == \"$FORMAT_MODEL:latest\" or .name == \"$FORMAT_MODEL\")" > /dev/null 2>&1; then
        print_error "Model '$FORMAT_MODEL' not found in Ollama"
        print_status "  Available models:"
        curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | sed 's/^/    - /'
        print_error "  Run: ollama pull $FORMAT_MODEL"
        return 1
    fi
    
    # Read the transcription
    local transcription
    if ! transcription=$(cat "$text_file"); then
        print_error "Failed to read transcription file: $text_file"
        return 1
    fi
    
    local text_length=${#transcription}
    print_verbose "  Transcription length: $text_length characters"
    
    if [ $text_length -eq 0 ]; then
        print_error "Transcription file is empty"
        return 1
    fi
    
    # Create prompt for Ollama
    local prompt="You are a text formatting assistant. Format the following transcription into a well-structured Markdown document. Add appropriate headers, paragraphs, and formatting. Do not add any content that is not in the original transcription. Only restructure and format the existing text.

Transcription:
$transcription

Please provide the formatted Markdown:"
    
    local escaped_prompt
    escaped_prompt=$(echo "$prompt" | jq -Rs .)
    
    local start_time
    local end_time
    local duration
    start_time=$(date +%s)
    
    # Call Ollama API with better error handling
    print_verbose "  Calling Ollama API..."
    local response
    local curl_exit_code
    
    response=$(curl -s --max-time 600 -w "\n%{http_code}" http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$FORMAT_MODEL\",
            \"prompt\": $escaped_prompt,
            \"stream\": false
        }" 2>&1)
    curl_exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Check curl exit code
    if [ $curl_exit_code -ne 0 ]; then
        print_error "Ollama API request failed (curl exit code: $curl_exit_code)"
        case $curl_exit_code in
            6) print_error "  Could not resolve host (check if Ollama is running)" ;;
            7) print_error "  Failed to connect to Ollama (is it running on port 11434?)" ;;
            28) print_error "  Request timeout (took longer than 600s)" ;;
            *) print_error "  Check network connectivity and Ollama service" ;;
        esac
        return 1
    fi
    
    # Extract HTTP status code (last line)
    local http_code
    http_code=$(echo "$response" | tail -n1)
    local response_body
    response_body=$(echo "$response" | sed '$d')
    
    # Check HTTP status
    if [ "$http_code" != "200" ]; then
        print_error "Ollama API returned HTTP $http_code"
        print_verbose "  Response: $response_body"
        
        # Try to extract error message
        local error_msg
        error_msg=$(echo "$response_body" | jq -r '.error // empty' 2>/dev/null)
        if [ -n "$error_msg" ]; then
            print_error "  API Error: $error_msg"
        fi
        return 1
    fi
    
    # Extract the formatted text from JSON response
    print_verbose "  Parsing Ollama response..."
    if ! echo "$response_body" | jq -e '.response' > /dev/null 2>&1; then
        print_error "Invalid JSON response from Ollama"
        print_verbose "  Response preview: $(echo "$response_body" | head -c 200)..."
        
        # Check if response contains error
        local error_msg
        error_msg=$(echo "$response_body" | jq -r '.error // empty' 2>/dev/null)
        if [ -n "$error_msg" ]; then
            print_error "  Error from Ollama: $error_msg"
        fi
        return 1
    fi
    
    # Save formatted output
    if ! echo "$response_body" | jq -r '.response' > "$md_file"; then
        print_error "Failed to write formatted output to: $md_file"
        return 1
    fi
    
    # Verify output file was created and is not empty
    if [ ! -s "$md_file" ]; then
        print_error "Output file is empty or was not created"
        return 1
    fi
    
    print_status "  ✓ Formatting complete (took ${duration}s)"
    return 0
}

format_text() {
    local text_file="$1"
    local md_file="$2"
    
    if [ "$USE_SIMPLE_FORMAT" = true ]; then
        format_text_simple "$text_file" "$md_file"
        return
    fi
    
    # Try LLM formatting first
    if ! format_text_llm "$text_file" "$md_file"; then
        print_warning "LLM formatting failed, falling back to simple formatting"
        format_text_simple "$text_file" "$md_file"
    fi
}

process_video() {
    local video_file="$1"
    
    echo ""
    print_status "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_status "Processing: $(basename "$video_file")"
    print_status "Location: $(dirname "$video_file")"
    
    # Determine output file paths
    local dir
    local filename
    local basename
    dir=$(dirname "$video_file")
    filename=$(basename "$video_file")
    basename="${filename%.*}"
    
    local audio_file="$dir/$basename.aac"
    local text_file="$dir/$basename.txt"
    local md_file="$dir/$basename.md"
    
    # Check if we should skip transcoding
    if [ "$SKIP_TRANSCODING" = true ] && [ -f "$text_file" ]; then
        print_status "✓ Transcription file exists: $basename.txt"
        print_status "  Skipping audio extraction and transcription (--skip-transcoding)"
        
        # Only do formatting
        if ! format_text "$text_file" "$md_file"; then
            ((FAILED_VIDEOS++))
            return 1
        fi
        
        print_status "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_status "✓✓✓ COMPLETED: $(basename "$video_file") (formatting only)"
        print_status "    Generated: $basename.md"
        ((PROCESSED_VIDEOS++))
        return 0
    fi
    
    # Check if video has audio track
    if ! has_audio_track "$video_file"; then
        print_warning "No audio track found, skipping"
        ((SKIPPED_VIDEOS++))
        return 0
    fi
    
    print_status "✓ Audio track detected"
    
    # Extract audio
    if ! extract_audio "$video_file" "$audio_file"; then
        ((FAILED_VIDEOS++))
        return 1
    fi
    
    # Transcribe audio
    if ! transcribe_audio "$audio_file" "$text_file"; then
        ((FAILED_VIDEOS++))
        return 1
    fi
    
    # Format text
    if ! format_text "$text_file" "$md_file"; then
        ((FAILED_VIDEOS++))
        return 1
    fi
    
    print_status "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_status "✓✓✓ COMPLETED: $(basename "$video_file")"
    print_status "    Generated: $basename.aac, $basename.txt, $basename.md"
    ((PROCESSED_VIDEOS++))
    return 0
}

################################################################################
# Main Functions
################################################################################

find_videos() {
    local root="$1"
    
    print_verbose "Searching for videos in: $root"
    
    # Build find command with all extensions
    local find_cmd="find \"$root\" -type f \\( "
    local first=true
    
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [ "$first" = true ]; then
            find_cmd+="-iname \"*.$ext\""
            first=false
        else
            find_cmd+=" -o -iname \"*.$ext\""
        fi
    done
    
    find_cmd+=" \\)"
    
    # Execute find and output results (one per line)
    eval "$find_cmd"
}

process_all_videos() {
    local root="$1"
    
    # Find all video files
    local videos=()
    while IFS= read -r video; do
        # Skip macOS resource fork files (._filename)
        if [[ $(basename "$video") == ._* ]]; then
            print_verbose "Skipping macOS resource fork: $video"
            continue
        fi
        videos+=("$video")
    done < <(find_videos "$root")
    
    TOTAL_VIDEOS=${#videos[@]}
    
    if [ $TOTAL_VIDEOS -eq 0 ]; then
        print_warning "No video files found in: $root"
        return 0
    fi
    
    print_status "Found $TOTAL_VIDEOS video file(s)"
    echo
    
    # Process each video
    local current=0
    for video in "${videos[@]}"; do
        ((current++))
        echo ""
        print_status "════════════════════════════════════════════════════════════════════"
        print_status "Video $current of $TOTAL_VIDEOS"
        print_status "════════════════════════════════════════════════════════════════════"
        process_video "$video"
    done
}

show_summary() {
    echo
    echo "======================================================================"
    print_status "Processing Summary"
    echo "======================================================================"
    echo "Total videos found:    $TOTAL_VIDEOS"
    echo "Successfully processed: $PROCESSED_VIDEOS"
    echo "Skipped (no audio):    $SKIPPED_VIDEOS"
    echo "Failed:                $FAILED_VIDEOS"
    echo "======================================================================"
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpu)
                USE_GPU=true
                shift
                ;;
            -c|--cpu)
                USE_GPU=false
                shift
                ;;
            -t|--transcription-model)
                TRANSCRIPTION_MODEL="$2"
                shift 2
                ;;
            -f|--format-model)
                FORMAT_MODEL="$2"
                shift 2
                ;;
            -s|--simple-format)
                USE_SIMPLE_FORMAT=true
                shift
                ;;
            --skip-transcoding)
                SKIP_TRANSCODING=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                ROOT_DIR="$1"
                shift
                ;;
        esac
    done
    
    # Check if directory is provided
    if [ -z "$ROOT_DIR" ]; then
        print_error "No directory specified"
        show_help
        exit 1
    fi
    
    # Check if directory exists
    if [ ! -d "$ROOT_DIR" ]; then
        print_error "Directory does not exist: $ROOT_DIR"
        exit 1
    fi
    
    # Convert to absolute path
    ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
}

################################################################################
# Main Entry Point
################################################################################

main() {
    echo "======================================================================"
    echo "Video/Audio Transcoding Script"
    echo "======================================================================"
    echo
    
    parse_arguments "$@"
    
    print_verbose "Configuration:"
    print_verbose "  Root directory: $ROOT_DIR"
    print_verbose "  GPU acceleration: $USE_GPU"
    print_verbose "  Transcription model: $TRANSCRIPTION_MODEL"
    print_verbose "  Format model: $FORMAT_MODEL"
    print_verbose "  Simple formatting: $USE_SIMPLE_FORMAT"
    print_verbose "  Skip transcoding: $SKIP_TRANSCODING"
    echo
    
    check_dependencies
    check_whisper_model
    check_ollama_model
    
    echo
    process_all_videos "$ROOT_DIR"
    
    show_summary
}

# Run main function
main "$@"
