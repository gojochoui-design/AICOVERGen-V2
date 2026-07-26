# AICoverGen Remaster V2

V2 replaces the old MDX-Net UVR models with **Mel-RoFormer** (becruily vocals + Sucial dereverb-echo) using `audio_separator` 0.44.5 vendored in the repo. Runs on **Google Colab Free**.

## Quick start

1. Open `AICoverGen_Remaster_Colab.ipynb` in Google Colab.
2. Run the single cell.
3. The notebook clones the repo, installs deps, downloads Mel-RoFormer models (~3 GB first run), and launches the WebUI.
4. Click the `public URL` that appears.
5. Upload a song + voice model, set pitch, click Generate.

See `README.md` for technical details, troubleshooting, and the full pipeline diagram.
