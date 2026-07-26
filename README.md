# 🎵 AICoverGen Remaster V2

<div align="center">

![Version](https://img.shields.io/badge/Version-2.0-mel_roformer?style=for-the-badge&color=blueviolet)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-lightgrey?style=for-the-badge&logo=googlecolab)

**AI-Powered Song Cover Generator · Applio Engine · Mel-RoFormer**

*Transform any song with any voice*

[🚀 Colab](#-quick-start) · [📖 How it works](#-how-it-works) · [🛠 Troubleshooting](#-troubleshooting)

</div>

---

## 🚀 Quick Start

1. Click the Colab badge below to open the notebook.
2. Run the single cell. It will clone the repo, install deps, download models (~3 GB first run) and launch the WebUI.
3. Click the `public URL` that appears.
4. Upload a song + voice model, set pitch, click Generate.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gojochoui-design/AICOVERGen-V2/blob/main/AICoverGen_Remaster_Colab.ipynb)

---

## 🧠 How it works

The pipeline runs 2 Mel-RoFormer separations + RVC voice conversion + final mix:

```
1. Song (mp3/wav/YouTube)
   │
   ▼  Mel-RoFormer becruily (vocals model)
2. vocals.wav  +  instrumental.wav
   │
   ▼  Mel-RoFormer Sucial (dereverb-echo model)
3. dry vocals (no reverb/echo)  +  No dry (residual)
   │
   ▼  Applio RVC engine
4. AI vocals (RVC voice)
   │
   ▼  Pedalboard (highpass + compressor + reverb)
5. AI vocals mixed (with natural ambiance)
   │
   ▼  Mix with instrumental
6. cover.mp3 🎵
```

### Models used

| Model | Author | Stems | Purpose |
|---|---|---|---|
| `mel_band_roformer_vocals_becruily.ckpt` | becruily | vocals + other | Separate vocals from instrumental |
| `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` | Sucial | dry + No dry | Strip reverb/echo from vocals (feeds RVC) |
| `mel_band_roformer_karaoke_becruily.ckpt` | becruily | Vocals + Instrumental | (cataloged, not used in pipeline) |

All models are downloaded automatically to `separator_models/` on first run, from the [`nomadkaraoke/python-audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) GitHub release.

The `audio_separator` 0.44.5 package is **vendored** in `audio_separator/` — no pip install needed for it.

---

## 🆕 What changed in V2 (vs V1)

V1 used 4 MDX-Net ONNX models from UVR. V2 replaces them with 2 Mel-RoFormer models:

| V1 (MDX-Net) | V2 (Mel-RoFormer) |
|---|---|
| `UVR-MDX-NET-Voc_FT.onnx` + `UVR-MDX-NET-Inst_HQ_4.onnx` | `mel_band_roformer_vocals_becruily.ckpt` |
| `UVR_MDXNET_KARA_2.onnx` + `Reverb_HQ_By_FoxJoy.onnx` | `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` |
| 4 MDX passes | 2 Mel-RoFormer passes (cleaner, faster) |
| `onnxruntime-gpu` (breaks on Colab CUDA 12) | torch directly (no onnxruntime needed) |
| `pip install audio_separator` | `audio_separator/` vendored in repo |

### Patches applied to vendored audio_separator

1. **ONNX Runtime optional** (`separator/separator.py`): `import onnxruntime` is wrapped in try/except. Mel-RoFormer models use torch, so onnxruntime is not needed.
2. **WAV export from MP3 input** (`separator/common_separator.py`): if the input is MP3 (subtype `MPEG_LAYER_III`), the WAV writer now uses `PCM_16` instead of propagating the invalid subtype.
3. **Version lookup graceful** (`separator/separator.py`): `get_package_distribution("audio-separator")` returns `None` when vendored, so the version string falls back to `"0.44.5-vendored"`.

---

## 📖 WebUI parameters

| Parameter | Default | What it does |
|---|---|---|
| Pitch Change | required | Semitones to shift AI voice. +12 for female-on-male, -12 for male-on-female. |
| Index Rate | 0.5 | How much to bias toward the dataset timbre (1) vs input (0). |
| Filter Radius | 3 | Median filter on detected pitch. Reduces breathiness. |
| RMS Mix Rate | 0.25 | How much to follow input loudness (0) vs fixed (1). |
| Protect | 0.33 | Protect voiceless consonants. 0.5 = disabled. |
| F0 Method | rmvpe | Pitch detection. `rmvpe` = clearer, `mangio-crepe` = smoother. |
| Reverb Size/Wet/Dry/Damping | 0.25 / 0.35 / 0.65 / 0.5 | Reverb applied to AI vocals after RVC. |
| Pitch Change All | 0 | Shift pitch of vocals AND instrumental together. |
| Keep Files | false | Keep intermediate WAVs (vocals, instrumental, etc.) for debugging. |

---

## 🗂 Repo structure

```
AICOVERGen-V2/
├── app.py                           # Entry point
├── AICoverGen_Remaster_Colab.ipynb  # Colab notebook (git clone mode)
├── requirements_colab.txt           # Python deps (audio_separator not listed: vendored)
├── packages.txt                     # apt packages: ffmpeg, sox, libsox-dev, libsndfile1
│
├── src/
│   ├── main.py                      # Pipeline: preprocess_song, song_cover_pipeline, combine_audio
│   ├── separator.py                 # Wrapper around audio_separator (run_separator)
│   ├── download_models.py           # Model download + validation
│   ├── webui.py                     # Gradio UI
│   ├── applio_adapter.py            # Applio RVC adapter
│   └── applio_worker.py             # RVC inference worker
│
├── audio_separator/                 # ⭐ audio_separator 0.44.5 vendored (with 3 patches)
│   ├── separator/
│   │   ├── separator.py             #   Separator class
│   │   ├── common_separator.py      #   Common logic
│   │   ├── architectures/
│   │   │   └── mdxc_separator.py    #   Mel-RoFormer architecture (the one we use)
│   │   └── uvr_lib_v5/              #   Neural net code (roformer, demucs, vr, mdx)
│   ├── models.json                  #   Supported models catalog
│   └── ...
│
├── separator_models/                # Model weights (auto-downloaded, gitignored)
├── rvc/                             # Applio RVC engine
├── rvc_models/                      # User-uploaded RVC models
└── song_output/                     # Generated covers
```

---

## 🛠 Troubleshooting

### `ImportError: libcudart.so.13`

Colab's preinstalled `onnxruntime-gpu` requires CUDA 13 but Colab only has CUDA 12. The notebook uninstalls onnxruntime at the start, so this should not happen. If it does, make sure you're running the latest notebook from the `main` branch.

### `audio_separator no produjo ningún stem`

Usually a WAV subtype issue. Make sure the patch to `common_separator.py` is applied (it converts MP3 subtypes to PCM_16 for WAV output).

### Duplicate voice in cover

If you hear both the original singer and the RVC voice, the backup vocals are leaking. The V2 pipeline sets `backup_vocals_path = None` to avoid this. Do not revert to using `vocals_path` as backup.

### Low quality / "doesn't sound like the original"

RVC always degrades quality somewhat. To minimize:
- Use a high-quality RVC model trained on clean vocals.
- Set `Index Rate` to 0.5–0.7 for better timbre match.
- Use `rmvpe` as the F0 method (clearer than `mangio-crepe`).
- Increase `Reverb Wet` to 0.35–0.45 for more natural ambiance.
- Avoid `Pitch Change All` (it slightly reduces quality).

### Models fail to download

Models are hosted on GitHub releases. If GitHub is slow/down, you can download them manually from `https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs` and put them in `separator_models/`.

---

## 📜 License

MIT. See `LICENSE`.

The vendored `audio_separator/` package is also MIT, (c) Andrew Beveridge — [original repo](https://github.com/karaokenerds/python-audio-separator).

Mel-RoFormer models are courtesy of their respective authors: **becruily** and **Sucial**.

---

## 🙏 Credits

- **AICoverGen** original: [SociallyIneptWeeb/AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)
- **Applio** (RVC engine): [ApplioTeam/Applio](https://github.com/ApplioTeam/Applio)
- **audio_separator**: [karaokenerds/python-audio-separator](https://github.com/karaokenerds/python-audio-separator)
- **Mel-RoFormer models**: becruily, Sucial, and the UVR community
