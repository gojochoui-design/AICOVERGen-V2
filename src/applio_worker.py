"""Worker CLI para ejecutar inferencia Applio en un proceso aislado."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _patch_torch_for_cpu_if_requested() -> None:
    """Fuerza CPU antes de importar Applio cuando el fallback lo solicita."""
    if os.environ.get("AICOVERGEN_FORCE_CPU") != "1":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Inferencia Applio aislada para AICoverGen Remaster")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--volume-envelope", type=float, default=0.8)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--split-audio", action="store_true")
    parser.add_argument("--clean-audio", action="store_true")
    parser.add_argument("--clean-strength", type=float, default=0.5)
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    _patch_torch_for_cpu_if_requested()

    from rvc.infer.infer import VoiceConverter

    converter = VoiceConverter()
    converter.convert_audio(
        audio_input_path=args.input,
        audio_output_path=args.output,
        model_path=args.model,
        index_path=args.index or "",
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        volume_envelope=args.volume_envelope,
        protect=args.protect,
        hop_length=args.hop_length,
        split_audio=args.split_audio,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
        embedder_model="contentvec",
        embedder_model_custom=None,
        clean_audio=args.clean_audio,
        clean_strength=args.clean_strength,
        export_format="WAV",
        post_process=False,
        resample_sr=0,
        sid=0,
    )
    _release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
