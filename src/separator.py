"""Wrapper sobre `audio_separator` (Mel-RoFormer / BS-RoFormer / MDX / Demucs / VR).

Este módulo sustituye por completo al antiguo `src/mdx.py`. Ya no se usan modelos
MDX-Net de UVR; en su lugar se utilizan modelos Mel-RoFormer:

* `mel_band_roformer_vocals_becruily.ckpt`  → separa **Vocals / Instrumental**.
  En el proyecto se le llama "Deux" porque produce los dos stems en una sola
  pasada. Sustituye a los antiguos `UVR-MDX-NET-Voc_FT.onnx` y `UVR-MDX-NET-Inst_HQ_4.onnx`.
* `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt`  → quita **reverb y echo**
  de un stem vocal. Stems: `dry` (limpio) + `No dry` (residuo con las colas de
  reverb/eco). Sustituye a los antiguos `UVR_MDXNET_KARA_2.onnx` (separación
  main/backup) y `Reverb_HQ_By_FoxJoy.onnx` (de-reverb).

NOTA sobre el modelo "Kar" de becruily: el catálogo incluye también
`mel_band_roformer_karaoke_becruily.ckpt` (BECRILY_KARAOKE_MODEL), pero su YAML
tiene `instruments: [Vocals, Instrumental]`, así que se comporta igual que el
"Deux" y NO quita reverb. El pipeline de AICoverGen NO lo usa para el paso de
de-reverb; usa en su lugar el modelo de Sucial. Se deja en el catálogo por
completitud y por si en el futuro se quiere usar como segunda opinión para la
separación vocals/instrumental.

La función pública principal es `run_separator()`, con una firma intencionadamente
parecida a la del viejo `run_mdx()` para que el resto del proyecto apenas cambie.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch

# Aseguramos que el directorio raíz del repo esté en sys.path para poder
# importar el paquete vendoreado `audio_separator` aunque el cwd sea otro.
BASE_DIR = Path(__file__).resolve().parent.parent
import sys

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from audio_separator.separator.separator import Separator  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Constantes del proyecto
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR_MODELS_DIR = os.path.join(BASE_DIR, "separator_models")
os.makedirs(SEPARATOR_MODELS_DIR, exist_ok=True)

# Catálogo de modelos que usa este proyecto. Añadir aquí cualquier modelo nuevo
# que se quiera usar con `run_separator(model_key=...)`.
#
# IMPORTANTE: los campos `primary_stem` y `secondary_stem` deben coincidir
# EXACTAMENTE con los nombres del campo `instruments:` del YAML del modelo
# (respetando mayúsculas/minúsculas). El modelo escribe los stems con esos
# nombres exactos, y `run_separator()` los busca en el nombre del archivo de
# salida para renombrarlo al sufijo que pida el pipeline.
SEPARATOR_MODELS = {
    # "Deux" — sustituye a UVR-MDX-NET-Voc_FT y UVR-MDX-NET-Inst_HQ_4.
    # Modelo Mel-RoFormer de becruily que separa la canción en 2 stems:
    #   * vocals (primary)  → voces aisladas
    #   * other  (secondary) → pista instrumental
    # El YAML declara `instruments: [vocals, other]` (minúsculas, "other"
    # en vez de "Instrumental"). Por eso aquí usamos esos nombres exactos.
    "becruily_vocals": {
        "filename": "mel_band_roformer_vocals_becruily.ckpt",
        "config": "config_mel_band_roformer_vocals_becruily.yaml",
        "primary_stem": "vocals",
        "secondary_stem": "other",
    },
    # "Kar" / De-Reverb — sustituye a UVR_MDXNET_KARA_2 y a Reverb_HQ_By_FoxJoy.
    #
    # NOTA IMPORTANTE: becruily no tiene un modelo específico de de-reverb/echo.
    # El modelo "mel_band_roformer_karaoke_becruily.ckpt" que se llamaba "Kar"
    # en realidad separa Vocals/Instrumental igual que el "Deux" (su YAML tiene
    # `instruments: [Vocals, Instrumental]`), por lo que NO sirve para quitar
    # reverb. Lo dejamos en el catálogo por si alguien quiere usarlo, pero el
    # pipeline de AICoverGen usa en su lugar el modelo De-Reverb-Echo de Sucial,
    # que sí está entrenado para esto y produce:
    #   * dry (primary)    → voz limpia sin reverb/echo
    #   * No dry (secondary) → residuo con las colas de reverberación y eco
    "becruily_karaoke": {
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
        "config": "config_mel_band_roformer_karaoke_becruily.yaml",
        "primary_stem": "Vocals",
        "secondary_stem": "Instrumental",
    },
    # De-Reverb-Echo por Sucial — modelo Mel-RoFormer entrenado para quitar
    # reverb Y eco de las vocales. Stems: dry (limpio) + No dry (residuo).
    # Este es el que usa el pipeline de AICoverGen como paso de "dereverb".
    "sucial_dereverb_echo": {
        "filename": "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
        "config": "config_dereverb-echo_mel_band_roformer.yaml",
        "primary_stem": "dry",
        "secondary_stem": "No dry",
    },
}

# Atajos para `main.py`.
# Mantenemos BECRILY_KARAOKE_MODEL por compatibilidad, pero el pipeline ya NO
# lo usa para el paso de de-reverb (usa SUCIAL_DEREVERB_ECHO_MODEL en su lugar).
BECRILY_VOCALS_MODEL = "becruily_vocals"
BECRILY_KARAOKE_MODEL = "becruily_karaoke"
SUCIAL_DEREVERB_ECHO_MODEL = "sucial_dereverb_echo"

# Tamaños mínimos razonables (en bytes) para considerar que el modelo está
# completo. Si el archivo en disco es más pequeño, se fuerza re-descarga.
SEPARATOR_EXPECTED_MIN_SIZES = {
    "mel_band_roformer_vocals_becruily.ckpt": 30_000_000,
    "mel_band_roformer_karaoke_becruily.ckpt": 30_000_000,
    "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt": 30_000_000,
}

# URLs de descarga. audio_separator las resuelve automáticamente al llamar a
# `Separator.load_model(...)`, pero las exponemos también para que el script
# `src/download_models.py` pueda descargarlas de forma explícita y validarlas.
MODEL_CONFIGS_REPO = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

SEPARATOR_DOWNLOAD_URLS = {
    "mel_band_roformer_vocals_becruily.ckpt": f"{MODEL_CONFIGS_REPO}/mel_band_roformer_vocals_becruily.ckpt",
    "config_mel_band_roformer_vocals_becruily.yaml": f"{MODEL_CONFIGS_REPO}/config_mel_band_roformer_vocals_becruily.yaml",
    "mel_band_roformer_karaoke_becruily.ckpt": f"{MODEL_CONFIGS_REPO}/mel_band_roformer_karaoke_becruily.ckpt",
    "config_mel_band_roformer_karaoke_becruily.yaml": f"{MODEL_CONFIGS_REPO}/config_mel_band_roformer_karaoke_becruily.yaml",
    "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt": f"{MODEL_CONFIGS_REPO}/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    "config_dereverb-echo_mel_band_roformer.yaml": f"{MODEL_CONFIGS_REPO}/config_dereverb-echo_mel_band_roformer.yaml",
}

# Logger del módulo — silencioso por defecto para no llenar la consola de Colab.
_logger = logging.getLogger("aicovergen.separator")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[separator] %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_model_key(model_key_or_filename: str) -> str:
    """Devuelve la clave canónica del catálogo a partir de una clave o del
    nombre de archivo del modelo. Lanza `KeyError` si no se reconoce."""
    if model_key_or_filename in SEPARATOR_MODELS:
        return model_key_or_filename
    for key, info in SEPARATOR_MODELS.items():
        if info["filename"] == model_key_or_filename:
            return key
    raise KeyError(
        f"Modelo de separación desconocido: {model_key_or_filename!r}. "
        f"Modelos disponibles: {list(SEPARATOR_MODELS.keys())}"
    )


def _build_separator(output_dir: str, base_device: str = "cuda", log_level: int = logging.INFO) -> Separator:
    """Crea una instancia fresca de `Separator` configurada para CPU o CUDA.

    audio_separator decide internamente el provider de ONNX Runtime, pero el
    dispositivo Torch lo fijamos nosotros para que coincida con lo que pide
    el resto del proyecto.
    """
    use_cuda = base_device.lower().startswith("cuda") and torch.cuda.is_available()

    # Parámetros por defecto para MDXC (los Mel-RoFormer se cargan como MDXC).
    # `segment_size=256` y `overlap=8` son los defaults recomendados por
    # audio_separator para modelos Mel-RoFormer.
    mdxc_params = {
        "segment_size": 256,
        "override_model_segment_size": False,
        "batch_size": 1,
        "overlap": 8,
        "pitch_shift": 0,
    }

    separator = Separator(
        log_level=log_level,
        model_file_dir=SEPARATOR_MODELS_DIR,
        output_dir=output_dir,
        output_format="WAV",
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        sample_rate=44100,
        use_soundfile=True,
        use_autocast=use_cuda,
        mdxc_params=mdxc_params,
    )

    if use_cuda:
        # Forzamos CUDA explícitamente; audio_separator a veces cae a CPU si
        # detecta simultáneamente onnxruntime-cpu y onnxruntime-gpu.
        separator.torch_device = torch.device("cuda:0")
        try:
            separator.onnx_execution_provider = ["CUDAExecutionProvider"]
        except Exception:
            pass

    return separator


def _rename_output(actual_path: str, target_basename: str) -> str:
    """Renombra el archivo de salida producido por audio_separator al basename
    que el resto del proyecto espera (por ejemplo `cancion_Vocals.wav`).

    Si el archivo ya tiene ese nombre, no hace nada. Si existe un archivo
    previo con el nombre destino, lo sobrescribe.
    """
    target_dir = os.path.dirname(actual_path)
    target_path = os.path.join(target_dir, target_basename)
    if os.path.abspath(actual_path) == os.path.abspath(target_path):
        return target_path
    try:
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(actual_path, target_path)
    except OSError as exc:
        _logger.warning("No se pudo renombrar %s → %s: %s", actual_path, target_path, exc)
        return actual_path
    return target_path


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

def ensure_separator_models_ready() -> list:
    """Comprueba que los dos modelos Mel-RoFormer de becruily existen y no están
    corruptos. Si falta alguno, intenta descargarlo a través del propio
    `audio_separator.Separator` (que ya trae lógica de reintentos).

    Returns:
        list: Nombres de los modelos que NO se pudieron preparar (vacío = OK).
    """
    failed: list[str] = []

    for key, info in SEPARATOR_MODELS.items():
        model_filename = info["filename"]
        model_path = os.path.join(SEPARATOR_MODELS_DIR, model_filename)
        config_filename = info["config"]
        config_path = os.path.join(SEPARATOR_MODELS_DIR, config_filename)

        # 1) ¿Existe el archivo?
        missing = not os.path.exists(model_path) or not os.path.exists(config_path)

        # 2) ¿Es lo bastante grande? (sólo si existe)
        too_small = False
        if not missing:
            min_size = SEPARATOR_EXPECTED_MIN_SIZES.get(model_filename, 0)
            if min_size > 0 and os.path.getsize(model_path) < min_size:
                too_small = True

        if missing or too_small:
            reason = "missing" if missing else f"too small ({os.path.getsize(model_path) / 1e6:.1f}MB)"
            _logger.info("Modelo %s está %s. Descargando...", model_filename, reason)
            try:
                # `load_model` descarga el .ckpt y su .yaml automáticamente
                # usando el catálogo interno de audio_separator.
                tmp_sep = _build_separator(output_dir=SEPARATOR_MODELS_DIR, base_device="cpu", log_level=logging.INFO)
                tmp_sep.load_model(model_filename)
                # liberamos memoria del modelo cargado (no lo vamos a usar ahora)
                del tmp_sep
                gc.collect()
            except Exception as exc:
                _logger.error("Falló la descarga de %s: %s", model_filename, exc)
                failed.append(model_filename)
                continue

        # Re-validar tras la descarga
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            _logger.info("Modelo OK: %s (%.1f MB)", model_filename, size_mb)
        else:
            failed.append(model_filename)

    return failed


def run_separator(
    model_key: str,
    output_dir: str,
    input_path: str,
    *,
    primary_suffix: Optional[str] = None,
    secondary_suffix: Optional[str] = None,
    exclude_primary: bool = False,
    exclude_secondary: bool = False,
    keep_orig: bool = True,
    base_device: str = "cuda",
    log_level: int = logging.INFO,
) -> Tuple[Optional[str], Optional[str]]:
    """Ejecuta la separación de un archivo de audio con un modelo Mel-RoFormer.

    Args:
        model_key: Clave del catálogo (`becruily_vocals`, `becruily_karaoke`)
            o el nombre de archivo del modelo.
        output_dir: Carpeta donde escribir los stems de salida.
        input_path: Ruta del wav de entrada.
        primary_suffix: Sufijo para el stem primario. Por ejemplo `"Vocals"`
            produce `cancion_Vocals.wav`. Si es `None`, se usa el nombre
            canónico del stem definido en el catálogo.
        secondary_suffix: Igual para el stem secundario.
        exclude_primary: Si es `True`, no se guarda el stem primario.
        exclude_secondary: Si es `True`, no se guarda el stem secundario.
        keep_orig: Si es `False`, se borra el archivo de entrada al terminar.
        base_device: `"cuda"` o `"cpu"`.

    Returns:
        (primary_path, secondary_path) — cada uno puede ser `None` si se
        excluyó o si el modelo no produce ese stem.
    """
    key = _resolve_model_key(model_key)
    info = SEPARATOR_MODELS[key]
    model_filename = info["filename"]
    primary_stem_name = info["primary_stem"]
    secondary_stem_name = info["secondary_stem"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo de entrada no existe: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1) ¿Modelo descargado? Si no, lo descargamos ahora mismo (mejor experiencia
    #    en Colab que un crash en mitad del pipeline).
    failed = ensure_separator_models_ready()
    if failed:
        raise RuntimeError(
            f"No se pudieron preparar los modelos del separador: {', '.join(failed)}. "
            "Ejecuta `python src/download_models.py` para descargarlos manualmente."
        )

    # 2) Configuramos qué stems queremos que escriba audio_separator.
    output_single_stem = None
    if exclude_primary and not exclude_secondary:
        output_single_stem = secondary_stem_name
    elif exclude_secondary and not exclude_primary:
        output_single_stem = primary_stem_name
    elif exclude_primary and exclude_secondary:
        # Caso degenerado: el usuario no quiere ninguno. No tiene sentido,
        # pero lo respetamos no haciendo nada.
        if not keep_orig and os.path.exists(input_path):
            os.remove(input_path)
        return None, None

    # 3) Construir el separador y cargar el modelo.
    separator = _build_separator(output_dir=output_dir, base_device=base_device, log_level=log_level)
    if output_single_stem is not None:
        separator.output_single_stem = output_single_stem

    separator.load_model(model_filename)

    # 4) Ejecutar la separación. audio_separator devuelve una lista con las
    #    rutas de los stems escritos (basename = `{audio_base}_({stem})_{model}.wav`).
    outputs = separator.separate(input_path)
    if not outputs:
        raise RuntimeError(f"audio_separator no produjo ningún stem para {input_path}")

    # 5) Mapear cada salida a su stem canónico y renombrar al sufijo pedido.
    primary_path: Optional[str] = None
    secondary_path: Optional[str] = None
    audio_base = os.path.splitext(os.path.basename(input_path))[0]

    for out_path in outputs:
        # audio_separator escribe en output_dir con basename relativo;
        # normalizamos a ruta absoluta.
        if not os.path.isabs(out_path):
            out_path = os.path.join(separator.output_dir, out_path)

        fname = os.path.basename(out_path)
        fname_lower = fname.lower()

        # audio_separator escribe los archivos como:
        #   {audio_base}_({stem_name})_{model_name}.wav
        # Hacemos match del stem name entre paréntesis para evitar falsos
        # positivos (p.ej. "vocals" aparecería tanto en "(vocals)" como en
        # cualquier nombre de archivo que mencionara "vocals").
        import re as _re
        stem_match = _re.search(r'_\(([^)]+)\)', fname_lower)
        stem_token = stem_match.group(1) if stem_match else fname_lower

        if stem_token == primary_stem_name.lower():
            target_suffix = primary_suffix if primary_suffix is not None else primary_stem_name
            target_basename = f"{audio_base}_{target_suffix}.wav"
            primary_path = _rename_output(out_path, target_basename)
        elif stem_token == secondary_stem_name.lower():
            target_suffix = secondary_suffix if secondary_suffix is not None else secondary_stem_name
            target_basename = f"{audio_base}_{target_suffix}.wav"
            secondary_path = _rename_output(out_path, target_basename)
        else:
            # Fallback: si el token entre paréntesis no casa exactamente con
            # ningún stem canónico, probamos con un `in` para ser más laxos.
            if primary_stem_name.lower() in stem_token:
                target_suffix = primary_suffix if primary_suffix is not None else primary_stem_name
                target_basename = f"{audio_base}_{target_suffix}.wav"
                primary_path = _rename_output(out_path, target_basename)
            elif secondary_stem_name.lower() in stem_token:
                target_suffix = secondary_suffix if secondary_suffix is not None else secondary_stem_name
                target_basename = f"{audio_base}_{target_suffix}.wav"
                secondary_path = _rename_output(out_path, target_basename)
            else:
                _logger.debug("Stem no esperado: %s (token=%r)", out_path, stem_token)

    # 6) Limpieza.
    del separator
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if not keep_orig and os.path.exists(input_path):
        try:
            os.remove(input_path)
        except OSError:
            pass

    return primary_path, secondary_path


def release_separator_memory() -> None:
    """Atajo para liberar memoria tras una separación intensiva."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "BASE_DIR",
    "SEPARATOR_MODELS_DIR",
    "SEPARATOR_MODELS",
    "SEPARATOR_DOWNLOAD_URLS",
    "SEPARATOR_EXPECTED_MIN_SIZES",
    "BECRILY_VOCALS_MODEL",
    "BECRILY_KARAOKE_MODEL",
    "SUCIAL_DEREVERB_ECHO_MODEL",
    "ensure_separator_models_ready",
    "run_separator",
    "release_separator_memory",
]
