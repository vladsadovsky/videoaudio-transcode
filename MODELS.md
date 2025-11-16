# Ollama Model Guide for Video Transcoding

Quick reference for selecting and downloading Ollama models based on your GPU.

## Model Selection by VRAM

### 24GB+ VRAM (RTX 3090 Ti, RTX 4090, A5000, etc.)

**Best models:**
```bash
ollama pull qwen2.5:32b      # 20GB - Highest quality
ollama pull llama3.1:13b     # 7.4GB - Very high quality
ollama pull qwen2.5:14b      # 9GB - Excellent quality
ollama pull qwen2.5:7b       # 5GB - Fallback
```

**Auto-selected:** qwen2.5:32b or qwen2.5:14b

---

### 16-24GB VRAM (RTX 4080, RTX 3080 Ti, A4000)

**Best models:**
```bash
ollama pull qwen2.5:14b      # 9GB - Primary
ollama pull llama3.1:13b     # 7.4GB - Alternative
ollama pull qwen2.5:7b       # 5GB - Fallback
ollama pull mistral          # 4.4GB - Extra fallback
```

**Auto-selected:** qwen2.5:14b

---

### 12-16GB VRAM (RTX 3060 12GB, RTX 4060 Ti 16GB)

**Best models:**
```bash
ollama pull qwen2.5:14b      # 9GB - Primary (tight fit with Whisper)
ollama pull qwen2.5:7b       # 5GB - Safer fallback
ollama pull mistral          # 4.4GB - Extra fallback
```

**Auto-selected:** qwen2.5:14b (with auto-fallback to qwen2.5:7b if needed)

**Note:** 14B model works but is tight. Script reserves 2GB for Whisper, leaving ~10GB for LLM.

---

### 8-12GB VRAM (RTX 3060 Ti, RTX 2080 Ti, RTX 4060)

**Best models:**
```bash
ollama pull qwen2.5:7b       # 5GB - Primary (recommended)
ollama pull mistral          # 4.4GB - Alternative
ollama pull phi3:mini        # 3GB - Fallback
```

**Auto-selected:** qwen2.5:7b

---

### 6-8GB VRAM (RTX 3060, RTX 2060, GTX 1070 Ti)

**Best models:**
```bash
ollama pull qwen2.5:7b       # 5GB - Primary
ollama pull phi3:mini        # 3GB - Fallback
ollama pull gemma2:2b        # 2GB - Safe fallback
```

**Auto-selected:** qwen2.5:7b or phi3:mini

---

### 4-6GB VRAM (GTX 1060 6GB, RTX 3050)

**Best models:**
```bash
ollama pull phi3:mini        # 3GB - Primary
ollama pull gemma2:2b        # 2GB - Fallback
```

**Auto-selected:** phi3:mini

---

### <4GB VRAM (GTX 1050 Ti, older cards)

**Best models:**
```bash
ollama pull gemma2:2b        # 2GB - Only option
```

**Auto-selected:** gemma2:2b

**Alternative:** Use `--cpu` flag or `--simple-format` to avoid LLM entirely.

---

## Model Quality Comparison

| Model         | Size  | Quality Score | Speed  | Best For                    |
|---------------|-------|---------------|--------|-----------------------------|
| qwen2.5:32b   | 20GB  | 100/100       | Slow   | 24GB+ VRAM, best quality    |
| qwen2.5:14b   | 9GB   | 95/100        | Medium | 12GB+ VRAM, excellent       |
| llama3.1:13b  | 7.4GB | 92/100        | Medium | 16GB+ VRAM, high quality    |
| qwen2.5:7b    | 5GB   | 90/100        | Fast   | **Recommended default**     |
| mistral       | 4.4GB | 85/100        | Fast   | 8GB+ VRAM, good quality     |
| llama3.1:8b   | 4.7GB | 88/100        | Fast   | 8GB+ VRAM, alternative      |
| phi3:mini     | 3GB   | 78/100        | Fast   | 4-6GB VRAM, efficient       |
| gemma2:2b     | 2GB   | 70/100        | Fast   | Limited VRAM, compact       |

## Download Commands

### Check GPU VRAM

```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

### Check Downloaded Models

```bash
ollama list
```

### Download Single Model

```bash
ollama pull qwen2.5:7b
```

### Download Multiple Models (for fallback)

```bash
# For 12GB VRAM system (recommended)
ollama pull qwen2.5:14b &
ollama pull qwen2.5:7b &
wait
```

### Remove Unwanted Models

```bash
ollama rm llama2  # Remove if not needed
```

## Installation Script Auto-Download

The `./install.sh` script automatically downloads optimal models based on detected VRAM:

```bash
./install.sh
# Automatically downloads:
# - Primary model for your VRAM
# - 1-3 fallback models
# - Shows progress for each download
```

## During Transcoding

The Python script automatically:
1. Tests each downloaded model
2. Selects the best one that works
3. Falls back to smaller models if needed
4. Shows which model is being used in verbose mode

```bash
# See model selection in action
./transcode.py -v /path/to/videos
```

## Manual Override

Force specific model:
```bash
./transcode.py -f qwen2.5:14b /path/to/videos
```

Use CPU-only (no Ollama):
```bash
./transcode.py --simple-format /path/to/videos
```

## Troubleshooting

**Model fails to load (memory error):**
```bash
# Solution 1: Let script auto-fallback
./transcode.py /path/to/videos  # Auto-tries smaller models

# Solution 2: Use CPU for Whisper, GPU for LLM
./transcode.py --cpu -f qwen2.5:14b /path/to/videos

# Solution 3: Force smaller model
./transcode.py -f phi3:mini /path/to/videos

# Solution 4: Skip LLM entirely
./transcode.py --simple-format /path/to/videos
```

**Download is slow:**
- Models are large (2-20GB)
- First download takes 5-30 minutes depending on model size
- Subsequent uses are instant

**Check if model is actually loaded:**
```bash
# Start Ollama server
ollama serve

# Test model
ollama run qwen2.5:7b "Hello"
```

## Quick Reference

**Your system: 12GB VRAM**

**Recommended setup:**
```bash
ollama pull qwen2.5:14b   # Primary (9GB) - optimal quality
ollama pull qwen2.5:7b    # Fallback (5GB) - safer option
```

**Then just run:**
```bash
./transcode.py /path/to/videos
# Script auto-selects qwen2.5:14b
# Auto-falls back to qwen2.5:7b if needed
```

**Check selection:**
```bash
./transcode.py -v /path/to/videos
# Shows: "Auto-detecting best Ollama model for your GPU..."
# Shows: "Selected model: qwen2.5:14b (optimal for 12.0 GB VRAM)"
```
