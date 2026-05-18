"""Adaptador de inferencia Applio para AICoverGen Remaster.

Este módulo reemplaza el motor RVC antiguo de AICoverGen por el motor moderno de
Applio, manteniendo una API pequeña y robusta para `src/main.py`.

Diseño para Colab Free:
- La inferencia se ejecuta en un subproceso para liberar VRAM al terminar.
- Se activa `split_audio=True` para reducir picos de memoria en audios largos.
- Si CUDA falla por OOM u otro error de GPU, se limpia memoria y se reintenta en CPU.
- No se carga HuBERT globalmente al importar el proyecto.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - Colab instalará torch si falta
    torch = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent


def _is_cuda_related_error(text: str) -> bool:
    text_l = text.lower()
    markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cublas",
        "cudnn",
        "cuda error",
        "illegal memory access",
        "device-side assert",
        "misaligned address",
        "no kernel image",
    )
    return any(m in text_l for m in markers)


def release_memory() -> None:
    """Libera memoria Python y caché CUDA si está disponible."""
    gc.collect()
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_worker(
    *,
    input_path: str,
    output_path: str,
    model_path: str,
    index_path: str,
    pitch_change: int,
    f0_method: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
    split_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    cpu: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["AICOVERGEN_FORCE_CPU"] = "1" if cpu else "0"
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        "-m",
        "src.applio_worker",
        "--input",
        input_path,
        "--output",
        output_path,
        "--model",
        model_path,
        "--index",
        index_path or "",
        "--pitch",
        str(int(pitch_change)),
        "--f0-method",
        str(f0_method),
        "--index-rate",
        str(float(index_rate)),
        "--volume-envelope",
        str(float(rms_mix_rate)),
        "--protect",
        str(float(protect)),
        "--hop-length",
        str(int(crepe_hop_length)),
        "--clean-strength",
        str(float(clean_strength)),
    ]
    if split_audio:
        cmd.append("--split-audio")
    if clean_audio:
        cmd.append("--clean-audio")

    return subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def rvc_infer_applio(
    *,
    index_path: str,
    index_rate: float,
    input_path: str,
    output_path: str,
    pitch_change: int,
    f0_method: str,
    model_path: str,
    rms_mix_rate: float = 0.8,
    protect: float = 0.5,
    crepe_hop_length: int = 128,
    split_audio: bool = True,
    clean_audio: bool = False,
    clean_strength: float = 0.5,
    allow_cpu_fallback: bool = True,
) -> str:
    """Convierte una pista vocal usando Applio RVC y devuelve la ruta de salida.

    La función conserva nombres parecidos a AICoverGen para integrarse con el
    flujo existente, pero delega todo el procesamiento vocal en Applio.
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    model_path = str(Path(model_path).resolve())
    index_path = str(Path(index_path).resolve()) if index_path else ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    release_memory()

    first = _run_worker(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        index_path=index_path,
        pitch_change=pitch_change,
        f0_method=f0_method,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        crepe_hop_length=crepe_hop_length,
        split_audio=split_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        cpu=False,
    )

    if first.returncode == 0 and Path(output_path).exists():
        release_memory()
        return output_path

    combined_log = f"STDOUT:\n{first.stdout}\nSTDERR:\n{first.stderr}"
    release_memory()

    if allow_cpu_fallback and _is_cuda_related_error(combined_log):
        print("[!] Falló la inferencia CUDA/VRAM; reintentando automáticamente en CPU.")
        second = _run_worker(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            index_path=index_path,
            pitch_change=pitch_change,
            f0_method=f0_method,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            crepe_hop_length=crepe_hop_length,
            split_audio=split_audio,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            cpu=True,
        )
        if second.returncode == 0 and Path(output_path).exists():
            release_memory()
            return output_path
        combined_log += f"\n\nCPU FALLBACK STDOUT:\n{second.stdout}\nCPU FALLBACK STDERR:\n{second.stderr}"

    raise RuntimeError("Applio RVC no pudo convertir el audio.\n" + combined_log[-6000:])
