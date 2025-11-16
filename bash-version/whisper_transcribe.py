#!/usr/bin/env python3
"""
Simple wrapper script for OpenAI Whisper transcription
Used by transcode.sh to transcribe audio files
"""

import sys
import os
import argparse

# Try to import whisper
try:
    import whisper
    import torch
except ImportError as e:
    print(f"Error: Required Python package not found: {e}", file=sys.stderr)
    print("Please run ./install.sh to install dependencies", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Whisper')
    parser.add_argument('audio_file', help='Path to audio file to transcribe')
    parser.add_argument('output_file', help='Path to output text file')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use (default: base)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference (default: cpu)')
    parser.add_argument('--language', default=None,
                       help='Language of the audio (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Check if CUDA is available
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
        device = 'cpu'
    
    try:
        # Load model
        print(f"Loading Whisper model '{args.model}' on {device}...", file=sys.stderr)
        model = whisper.load_model(args.model, device=device)
        print(f"Model loaded successfully", file=sys.stderr)
        
        # Get audio duration for progress estimation
        import subprocess
        try:
            duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                          'format=duration', '-of', 
                          'default=noprint_wrappers=1:nokey=1', args.audio_file]
            duration_output = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL)
            duration = float(duration_output.decode().strip())
            minutes = int(duration / 60)
            seconds = int(duration % 60)
            print(f"Audio duration: {minutes}m {seconds}s", file=sys.stderr)
        except:
            pass
        
        # Transcribe
        print(f"Transcribing (this may take a while)...", file=sys.stderr)
        result = model.transcribe(
            args.audio_file,
            language=args.language,
            verbose=True  # Show progress during transcription
        )
        
        # Write output
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'].strip())
        
        print(f"✓ Transcription saved to: {args.output_file}", file=sys.stderr)
        
        # Show detected language if auto-detected
        if args.language is None and 'language' in result:
            print(f"Detected language: {result['language']}", file=sys.stderr)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
