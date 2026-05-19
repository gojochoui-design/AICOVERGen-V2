# 🎵 AICoverGen Remaster

<div align="center">

![Version](https://img.shields.io/badge/Version-1.1-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge)

**AI-Powered Song Cover Generator with Applio Engine**

*Transform any song with any voice*

[🚀 Quick Start](#-quick-start---google-colab) · [📖 Documentation](#-how-to-use-the-web-interface) · [💻 Local Install](#-local-installation-guide)

</div>

---

## 🎯 What This Does

<table>
<tr>
<td width="50%">

**Input**
- Any song (MP3, WAV, YouTube link)
- A voice model (RVC v2 format)

</td>
<td width="50%">

**Output**
- AI cover with custom voice
- Same melody, different singer

</td>
</tr>
</table>

> 💡 **Example**: Take a pop song → Apply a "cartoon character" voice model → Get a cover where that character sings the song

---

## ⚡ Quick Start - Google Colab

<div align="center">

### The Easiest Way - No Installation Required

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XqgpigrJ9mmmgOCRmevnrwmGBNBSY_1I?usp=sharing)

</div>

<table>
<tr>
<th align="center">Step</th>
<th align="center">Action</th>
</tr>
<tr>
<td align="center"><b>1</b></td>
<td>Click the Colab button above</td>
</tr>
<tr>
<td align="center"><b>2</b></td>
<td>Set runtime to <b>T4 GPU</b> (Runtime → Change runtime type)</td>
</tr>
<tr>
<td align="center"><b>3</b></td>
<td>RUN and wait</td>
</tr>
<tr>
<td align="center"><b>4</b></td>
<td>Wait 2-3 minutes for setup</td>
</tr>
<tr>
<td align="center"><b>5</b></td>
<td>Click the <code>.gradio.live</code> link that appears</td>
</tr>
<tr>
<td align="center"><b>6</b></td>
<td>Create your AI covers!</td>
</tr>
</table>

---

## 🎛️ How To Use The Web Interface

### 📥 Step 1: Get A Voice Model

<table>
<tr>
<th width="33%">Option A: Download</th>
<th width="33%">Option B: Upload</th>
<th width="33%">Option C: Public Index</th>
</tr>
<tr>
<td>

Go to **Download model** tab

Paste a model URL

Give it a name

Click **Download**

</td>
<td>

Go to **Upload model** tab

Upload ZIP file

Must contain `.pth` file

Optional `.index` file

</td>
<td>

Go to **Download model** tab

Click **Initialize** button

Browse and select

Auto-fills the form

</td>
</tr>
</table>

> 🔍 **Where to find models**: Search "RVC v2 models" on Google or HuggingFace

---

### 🎵 Step 2: Provide Your Song

| Method | Instructions |
|--------|-------------|
| 📁 **Upload File** | Click upload area → Select MP3, WAV, M4A, FLAC, etc. |
| 🎬 **YouTube Link** | Paste URL → Click "Process URL" |

---

### 🎤 Step 3: Select Voice Model

```
┌─────────────────────────────────────┐
│  Voice Models: [ your_model_name ▼ ] │  ← Click dropdown
│                                     │
│  [ Refresh Models 🔄 ]               │  ← Click if not showing
└─────────────────────────────────────┘
```

---

### 🎚️ Step 4: Adjust Pitch

<div align="center">

| Situation | Pitch Setting | Why |
|-----------|--------------|-----|
| 👩 Female model + 🎵 Male song | `+12` semitones | Raise voice to match higher key |
| 👨 Male model + 🎵 Female song | `-12` semitones | Lower voice to match lower key |
| 👩 Female model + 🎵 Female song | `0` | Same gender, no change needed |
| 👨 Male model + 🎵 Male song | `0` | Same gender, no change needed |

</div>

---

### ▶️ Step 5: Generate

<table>
<tr>
<td width="33%" align="center">

<b>1. Click Generate</b>

</td>
<td width="33%" align="center">

<b>2. Wait 2-5 min</b>

</td>
<td width="33%" align="center">

<b>3. Download</b>

</td>
</tr>
</table>

---

## ⚙️ Settings Explained

### 🔊 Voice Conversion Settings

| Setting | What It Does | Recommended |
|---------|-------------|-------------|
| **Index Rate** | How much the index file influences the output | `0.5` (balanced) |
| **Filter Radius** | Smooths pitch detection results | `3` |
| **RMS Mix Rate** | Loudness consistency | `0.25` |
| **Protect** | Keeps breath sounds natural | `0.33` |
| **F0 Method** | Pitch detection algorithm | `rmvpe` (best quality) |
| **Denoise** | Removes background noise | ✅ On |

### 🎚️ Audio Mixing Settings

| Setting | What It Does | Recommended |
|---------|-------------|-------------|
| **Main Vocals** | Volume of AI voice (dB) | `0` |
| **Backup Vocals** | Volume of backing vocals (dB) | `0` |
| **Music** | Volume of instruments (dB) | `0` |
| **Room Size** | Reverb space size | `0.15` |
| **Wetness** | Amount of reverb effect | `0.2` |
| **Output Format** | MP3 (small) or WAV (quality) | MP3 |

### 🎼 F0 Method Comparison

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| `rmvpe` | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | **Most cases (recommended)** |
| `fcpe` | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Great | Long audio files |
| `crepe` | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Excellent | Professional quality |
| `hybrid` | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | Best of both worlds |

---

## 🔧 How It Actually Works

<div align="center">

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Song   │ ──▶ │  Separate Audio │ ──▶ │  Convert Voice  │
│  (MP3/YouTube)│     │  Vocals/Inst/   │     │  Apply AI Model │
│              │     │  Backup/Reverb  │     │                 │
└──────────────┘     └─────────────────┘     └─────────────────┘
                                                     │
                                                     ▼
                                             ┌─────────────────┐
                                             │  Add Effects +  │
                                             │  Mix Together   │
                                             │                 │
                                             └─────────────────┘
                                                     │
                                                     ▼
                                             ┌─────────────────┐
                                             │  Your Cover!    │
                                             │  (MP3 or WAV)   │
                                             └─────────────────┘
```

</div>

### 📖 Detailed Pipeline Steps

1. **📥 Download/Load** - Get your audio from file or YouTube
2. **✂️ Vocals/Instrumental Split** - MDX-Net separates voice from music
3. **🎤 Main/Backup Split** - Isolate lead singer from harmonies
4. **🔇 DeReverb** - Remove echo for cleaner voice conversion
5. **🔄 Voice Conversion** - Applio RVC transforms the voice
6. **🎛️ Effects** - Add reverb, compression
7. **🔀 Mix** - Combine everything back together
8. **💾 Export** - Save your cover

---

## 📁 File Structure

```
aicovergen_remaster/
│
├── 📄 app.py ................... Main entry point
├── 📄 requirements_colab.txt .... Python dependencies
│
├── 📁 src/
│   ├── 📄 main.py ............... Core processing logic
│   ├── 📄 webui.py .............. Gradio interface
│   ├── 📄 mdx.py ................ Audio separation
│   ├── 📄 applio_adapter.py ..... Voice conversion bridge
│   └── 📄 download_models.py .... Model downloader
│
├── 📁 rvc_models/ ............... Your voice models here
├── 📁 song_output/ .............. Generated covers appear here
└── 📁 mdxnet_models/ ............ Separation models (auto-downloaded)
```

---

## 🏆 Tips For Best Results

<div align="center">

| ✅ Do This | ❌ Avoid This |
|-----------|--------------|
| Use 320kbps MP3 or WAV source | Low quality 128kbps files |
| Get models with `.index` files | Models without index |
| Match pitch correctly | Random pitch values |
| Keep songs under 5 min (Colab) | 10+ minute songs |
| Use `rmvpe` F0 method | Unknown F0 methods |

</div>

---

## ❌ Troubleshooting

### 🔴 "CUDA out of memory" Error

**What it means**: Your GPU ran out of VRAM

**Solutions**:
- The system should auto-fallback to CPU
- Close other GPU applications (games, video editors)
- Use a shorter audio file
- Process during off-peak times

### 🔴 Voice Sounds Robotic

**Possible causes & fixes**:

| Cause | Fix |
|-------|-----|
| Wrong pitch | Adjust by +1 or -1 until natural |
| No index file | Get a model with `.index` file |
| Low index rate | Increase to 0.75 |
| Bad model | Try a different voice model |

### 🔴 Model Not Showing Up

**Solutions**:
1. Click "Refresh Models" button
2. Check the model is in `rvc_models/YOUR_MODEL/`
3. Make sure there's a `.pth` file inside
4. Restart the application

### 🔴 Download Failed

**Solutions**:
- Check your internet connection
- Try a different browser
- Use a direct HuggingFace link
- Download manually and upload the ZIP

---

## 📜 Credits & License

<div align="center">

| Project | Link |
|---------|------|
| 🎵 Original AICoverGen | [GitHub](https://github.com/SociallyIneptWeeb/AICoverGen) |
| 🎤 Applio Engine | Open-source RVC implementation |
| ✂️ MDX-Net / UVR | Audio separation models |

</div>

**License: MIT**

---

<div align="center">

**Made with ❤️ for music creators**

</div>

---

# 💻 Local Installation Guide

<div align="center">

![Hammann](https://media.tenor.com/J0YH0FH8k0UAAAAC/azur-lane-hammann.gif)

![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=for-the-badge)
![Time](https://img.shields.io/badge/Time-15%20minutes-blue?style=for-the-badge)
![Type](https://img.shields.io/badge/Type-Step%20by%20Step-green?style=for-the-badge)

**Complete Guide to Install AICoverGen Remaster on Your Computer**

</div>

---

## 📋 Requirements

### 💻 Hardware Requirements

<div align="center">

| Component | 🔴 Minimum | 🟢 Recommended |
|-----------|------------|----------------|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | GTX 1060 (6GB) | RTX 3060 or better |
| **Storage** | 10 GB free | 20 GB SSD |
| **Internet** | Required for setup | Broadband recommended |

</div>

> ⚠️ **GPU Note**: Only NVIDIA GPUs with CUDA support work for acceleration. AMD/Intel GPUs and CPU-only mode work but are 3-5x slower.

### 🖥️ Software Requirements

| Software | Version | Required |
|----------|---------|----------|
| Python | 3.10 or 3.11 | ✅ Yes |
| FFmpeg | Latest | ✅ Yes |
| CUDA Toolkit | 11.8 or 12.x | GPU only |
| Git | Latest | Optional |

---

## 🚀 Installation Steps

### Step 1: Install Python

<table>
<tr>
<th>Windows</th>
<th>Linux</th>
<th>macOS</th>
</tr>
<tr>
<td>

1. Go to [python.org](https://www.python.org/downloads/)

2. Download **Python 3.10.x**

3. Run installer

4. ✅ Check **"Add Python to PATH"**

5. Click Install

</td>
<td>

```bash
sudo apt update
sudo apt install python3.10
sudo apt install python3.10-venv
sudo apt install python3-pip
```

</td>
<td>

```bash
brew install python@3.10
```

</td>
</tr>
</table>

**Verify installation:**
```bash
python --version
# Output: Python 3.10.x
```

---

### Step 2: Install FFmpeg

<table>
<tr>
<th>Windows</th>
<th>Linux</th>
<th>macOS</th>
</tr>
<tr>
<td>

1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)

2. Get Windows build from gyan.dev

3. Extract ZIP to `C:\ffmpeg`

4. Add to PATH:
   - Press `Win + R`
   - Type `sysdm.cpl`
   - Advanced → Environment Variables
   - Edit "Path"
   - Add `C:\ffmpeg\bin`

</td>
<td>

```bash
sudo apt update
sudo apt install ffmpeg
```

</td>
<td>

```bash
brew install ffmpeg
```

</td>
</tr>
</table>

**Verify installation:**
```bash
ffmpeg -version
```

---

### Step 3: Install CUDA (GPU Users Only)

> ⏭️ Skip this step if you don't have an NVIDIA GPU

**Check if NVIDIA driver is installed:**
```bash
nvidia-smi
```

If you see your GPU info, drivers are installed. If not, install NVIDIA drivers first.

**Install CUDA Toolkit:**

1. Go to [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your OS
3. Download CUDA 11.8 or 12.x
4. Run the installer

**Verify installation:**
```bash
nvcc --version
```

---

### Step 4: Setup Project Directory

```bash
# Create a folder for the project
mkdir ai-covers
cd ai-covers

# Extract your ZIP file here
# Or clone from repository:
# git clone https://github.com/YOUR_REPO/aicovergen_remaster.git

# Navigate into project
cd aicovergen_remaster
```

---

### Step 5: Create Virtual Environment

**Why?** Keeps dependencies isolated from your system Python.

```bash
# Create virtual environment
python -m venv venv
```

**Activate it:**

| Platform | Command |
|----------|---------|
| **Windows** | `venv\Scripts\activate` |
| **Linux/macOS** | `source venv/bin/activate` |

You should see `(venv)` before your prompt:
```
(venv) C:\aicovergen_remaster>
```

---

### Step 6: Install PyTorch

Choose based on your setup:

<table>
<tr>
<th>🟢 NVIDIA GPU (CUDA 11.8)</th>
<th>🟡 NVIDIA GPU (CUDA 12.1)</th>
<th>🔴 CPU Only</th>
</tr>
<tr>
<td>

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

</td>
<td>

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

</td>
<td>

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

</td>
</tr>
</table>

**Verify GPU detection (GPU users only):**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
- `True` = GPU detected ✅
- `False` = GPU not detected, reinstall with correct CUDA version

---

### Step 7: Install Dependencies

```bash
pip install -r requirements_colab.txt
```

This takes 2-5 minutes. You'll see many packages being downloaded.

**Linux users may need additional packages:**
```bash
sudo apt install libsndfile1 sox libsox-dev
```

---

### Step 8: Download AI Models

```bash
python src/download_models.py
```

This downloads approximately **500MB** of required models:

| Model | Size | Purpose |
|-------|------|---------|
| hubert_base.pt | ~190MB | Speech features |
| rmvpe.pt | ~55MB | Pitch detection |
| MDX-Net models | ~250MB | Audio separation |

**Wait for:**
```
[+] All models ready!
```

---

### Step 9: Run the Application

```bash
python app.py
```

Or with more options:

| Command | Purpose |
|---------|---------|
| `python app.py` | Local access only |
| `python src/webui.py --listen` | LAN access |
| `python src/webui.py --share` | Public internet link |

---

## 🎮 Running the App

### After Installation

Every time you want to use AICoverGen:

```bash
# 1. Navigate to project
cd aicovergen_remaster

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Run
python app.py
```

### Access the Interface

| Mode | URL |
|------|-----|
| Local | `http://localhost:7860` |
| LAN | `http://YOUR_LOCAL_IP:7860` |
| Public | Link shown in terminal |

---

## 🔧 Troubleshooting

### ❌ "python is not recognized"

**Problem**: Python command not found in terminal

**Solutions**:
1. Reinstall Python with "Add to PATH" checked
2. Restart your terminal/command prompt
3. Restart your computer
4. On Windows, try `py` instead of `python`

---

### ❌ "pip is not recognized"

**Problem**: pip command not found

**Solutions**:
```bash
# Use python -m pip instead
python -m pip install -r requirements_colab.txt
```

---

### ❌ "ffmpeg is not recognized"

**Problem**: FFmpeg not in PATH

**Solutions**:
1. Verify FFmpeg is installed
2. Add to system PATH correctly
3. Restart terminal after adding to PATH
4. Restart computer

---

### ❌ torch.cuda.is_available() returns False

**Problem**: PyTorch can't detect GPU

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA version matches PyTorch version
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

### ❌ CUDA out of memory

**Problem**: GPU ran out of VRAM

**Solutions**:
1. Close other GPU applications (games, video editors)
2. Use shorter audio files
3. Let the system fallback to CPU automatically
4. Reduce audio quality

---

### ❌ Installation is very slow

**Problem**: pip downloads taking forever

**Solutions**:
```bash
# Use a faster mirror
pip install -r requirements_colab.txt -i https://pypi.org/simple/
```

---

### ❌ Permission denied (Linux/macOS)

**Problem**: Can't write to directories

**Solutions**:
```bash
# Don't use sudo with pip in venv
# Instead, make sure you own the directory
sudo chown -R $USER:$USER aicovergen_remaster
```

---

## 💡 Tips & Tricks

### 🚀 Create a Desktop Shortcut

**Windows (`start.bat`):**
```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate
python app.py
pause
```

**Linux/macOS (`start.sh`):**
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
```
Then: `chmod +x start.sh`

---

### 📊 Performance Comparison

| Setup | Speed | Quality |
|-------|-------|---------|
| RTX 3080 (GPU) | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐⭐⭐ Best |
| GTX 1060 (GPU) | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Best |
| CPU (Ryzen 7) | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Best |
| CPU (Older) | ⭐ Very Slow | ⭐⭐⭐⭐⭐ Best |

---

### 🎯 Getting Voice Models

| Source | Link | Notes |
|--------|------|-------|
| HuggingFace | [huggingface.co](https://huggingface.co) | Search "RVC model" |
| Applio Discord | Community server | User-shared models |
| YouTube | Tutorial videos | Links in descriptions |

---

### ⚡ Quick Commands Reference

```bash
# Activate venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS

# Deactivate venv
deactivate

# Run app
python app.py

# Update models
python src/download_models.py

# Update dependencies
pip install -r requirements_colab.txt --upgrade
```

---

<div align="center">

![Hammann](https://media.tenor.com/J0YH0FH8k0UAAAAC/azur-lane-hammann.gif)

**Happy Cover Making! 🎵**

</div>
