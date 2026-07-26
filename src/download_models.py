"""Descarga y validación de modelos para AICoverGen Remaster.

Ya NO se usan modelos MDX-Net de UVR. En su lugar, este script descarga los dos
modelos Mel-RoFormer de becruily que usa el pipeline de separación:

* `mel_band_roformer_vocals_becruily.ckpt`  → separa Vocals / Instrumental.
* `mel_band_roformer_karaoke_becruily.ckpt`  → separa Lead (sin reverb) / Backing.

Los modelos se guardan en `separator_models/`, junto con sus archivos YAML de
configuración. El paquete `audio_separator` 0.44.5 está vendoreado en el repo,
así que no depende de pip para funcionar.
"""

from pathlib import Path
import os
import sys
import requests

# Aseguramos que el directorio raíz del proyecto esté en sys.path para poder
# importar `separator` (que a su vez importa el paquete vendoreado
# `audio_separator`).
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

# URLs de descarga (espejo oficial de audio-separator en GitHub).
MODEL_CONFIGS_REPO = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

separator_models_dir = BASE_DIR / "separator_models"
rvc_models_dir = BASE_DIR / "rvc_models"

separator_models_dir.mkdir(parents=True, exist_ok=True)
rvc_models_dir.mkdir(parents=True, exist_ok=True)

# Modelos Mel-RoFormer usados por el pipeline:
# 1) becruily vocals (separación Vocals/Instrumental — el "Deux")
# 2) Sucial dereverb-echo (quita reverb y eco de las vocales — sustituye al
#    "Kar" de becruily, que en realidad separa Vocals/Instrumental como el Deux)
# El modelo "karaoke_becruily" se descarga también por si se quiere usar como
# segunda opinión, pero el pipeline NO lo usa.
SEPARATOR_MODELS_TO_DOWNLOAD = [
    {
        "filename": "mel_band_roformer_vocals_becruily.ckpt",
        "url": f"{MODEL_CONFIGS_REPO}/mel_band_roformer_vocals_becruily.ckpt",
        "min_size": 30_000_000,  # ~870 MB
    },
    {
        "filename": "config_mel_band_roformer_vocals_becruily.yaml",
        "url": f"{MODEL_CONFIGS_REPO}/config_mel_band_roformer_vocals_becruily.yaml",
        "min_size": 1_000,  # ~1-2 KB
    },
    {
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
        "url": f"{MODEL_CONFIGS_REPO}/mel_band_roformer_karaoke_becruily.ckpt",
        "min_size": 30_000_000,  # ~1.6 GB
    },
    {
        "filename": "config_mel_band_roformer_karaoke_becruily.yaml",
        "url": f"{MODEL_CONFIGS_REPO}/config_mel_band_roformer_karaoke_becruily.yaml",
        "min_size": 1_000,
    },
    {
        "filename": "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
        "url": f"{MODEL_CONFIGS_REPO}/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
        "min_size": 30_000_000,  # ~80-100 MB
    },
    {
        "filename": "config_dereverb-echo_mel_band_roformer.yaml",
        "url": f"{MODEL_CONFIGS_REPO}/config_dereverb-echo_mel_band_roformer.yaml",
        "min_size": 1_000,
    },
]

RVC_EXPECTED_MIN_SIZES = {
    "hubert_base.pt": 180_000_000,  # ~190 MB
    "rmvpe.pt": 50_000_000,  # ~55 MB
}

MAX_RETRIES = 3


def dl_separator_file(filename: str, url: str, dir_name: Path, min_size: int = 0, force: bool = False) -> bool:
    """Descarga un archivo del separador (modelo .ckpt o config .yaml).

    Reintenta hasta MAX_RETRIES veces y valida el tamaño mínimo del archivo
    descargado. Devuelve True si el archivo está OK al final.
    """
    dest = dir_name / filename

    if not force and dest.exists():
        if min_size > 0 and dest.stat().st_size < min_size:
            print(f"[!] {filename} existe pero es demasiado pequeño "
                  f"({dest.stat().st_size / 1e6:.2f}MB < {min_size / 1e6:.2f}MB). Re-descargando...")
            dest.unlink()
        else:
            return True

    if force and dest.exists():
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Descargando {filename}... (intento {attempt}/{MAX_RETRIES})")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                actual_size = dest.stat().st_size

                if total_size > 0 and actual_size != total_size:
                    print(f"[!] {filename}: descargados {actual_size} bytes pero el servidor dijo {total_size}.")
                    if attempt < MAX_RETRIES:
                        dest.unlink()
                        continue
                    return False

                if min_size > 0 and actual_size < min_size:
                    print(f"[!] {filename}: el archivo descargado pesa {actual_size / 1e6:.2f}MB "
                          f"pero se esperaba al menos {min_size / 1e6:.2f}MB.")
                    if attempt < MAX_RETRIES:
                        dest.unlink()
                        continue
                    return False

                unit = "MB"
                size_str = f"{actual_size / 1e6:.2f} MB" if actual_size >= 1_000_000 else f"{actual_size} bytes"
                print(f"[+] {filename} descargado correctamente ({size_str})")
                return True

        except requests.exceptions.RequestException as e:
            print(f"[!] Error descargando {filename} (intento {attempt}/{MAX_RETRIES}): {e}")
            if dest.exists():
                dest.unlink()
            if attempt < MAX_RETRIES:
                import time
                time.sleep(2)
            else:
                return False

    return False


def dl_model(link, model_name, dir_name, force=False):
    """Compatibilidad hacia atrás: descarga un modelo RVC con la firma antigua.

    Se conserva para no romper posibles scripts externos. Internamente delega
    en `dl_separator_file` cuando el archivo está en el catálogo del separador;
    en caso contrario asume que es un modelo RVC y lo descarga con la URL base
    proporcionada.
    """
    # ¿Es un archivo del separador?
    for entry in SEPARATOR_MODELS_TO_DOWNLOAD:
        if entry["filename"] == model_name:
            return dl_separator_file(
                model_name, entry["url"], dir_name, min_size=entry["min_size"], force=force
            )

    # Si no, lo tratamos como modelo RVC (comportamiento histórico).
    model_path = dir_name / model_name
    expected_min = RVC_EXPECTED_MIN_SIZES.get(model_name, 0)

    if not force and model_path.exists():
        if expected_min > 0:
            actual_size = os.path.getsize(model_path)
            if actual_size >= expected_min:
                return True
            print(f"[!] {model_name} existe pero es demasiado pequeño "
                  f"({actual_size / 1e6:.2f}MB < {expected_min / 1e6:.2f}MB). Re-descargando...")
            os.remove(model_path)
        else:
            return True

    if force and model_path.exists():
        os.remove(model_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Descargando {model_name}... (intento {attempt}/{MAX_RETRIES})")
            with requests.get(f"{link}{model_name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                actual_size = os.path.getsize(model_path)
                if total_size > 0 and actual_size != total_size:
                    print(f"[!] {model_name}: descargados {actual_size} bytes, esperados {total_size}.")
                    if attempt < MAX_RETRIES:
                        os.remove(model_path)
                        continue
                    return False
                if expected_min > 0 and actual_size < expected_min:
                    print(f"[!] {model_name}: descargado {actual_size / 1e6:.2f}MB < {expected_min / 1e6:.2f}MB.")
                    if attempt < MAX_RETRIES:
                        os.remove(model_path)
                        continue
                    return False
                print(f"[+] {model_name} descargado correctamente ({actual_size / 1e6:.2f} MB)")
                return True
        except requests.exceptions.RequestException as e:
            print(f"[!] Error descargando {model_name} (intento {attempt}/{MAX_RETRIES}): {e}")
            if model_path.exists():
                os.remove(model_path)
            if attempt < MAX_RETRIES:
                import time
                time.sleep(2)
            else:
                return False
    return False


def validate_existing_models(dir_name, expected_sizes, link):
    """Valida modelos existentes y re-descarga los corruptos.

    Mantenida por compatibilidad con código externo; ignora silenciosamente
    archivos que no estén en `expected_sizes`.
    """
    invalid = []
    for model_name, min_size in expected_sizes.items():
        model_path = dir_name / model_name
        if model_path.exists():
            actual_size = os.path.getsize(model_path)
            if actual_size < min_size:
                print(f"[!] {model_name} corrupto ({actual_size / 1e6:.2f}MB < {min_size / 1e6:.2f}MB). Re-descargando...")
                os.remove(model_path)
                if not dl_model(link, model_name, dir_name):
                    invalid.append(model_name)
    return invalid


def download_separator_models() -> list:
    """Descarga todos los archivos (modelos + YAMLs) del separador.

    Returns:
        list: Nombres de los archivos que NO se pudieron descargar (vacío = OK).
    """
    failed = []
    for entry in SEPARATOR_MODELS_TO_DOWNLOAD:
        if not dl_separator_file(
            entry["filename"], entry["url"], separator_models_dir, min_size=entry["min_size"]
        ):
            failed.append(entry["filename"])
    return failed


if __name__ == "__main__":
    print("=" * 60)
    print("AICoverGen Remaster - Descarga de modelos (Mel-RoFormer becruily)")
    print("=" * 60)

    print("\n[1/2] Descargando modelos del separador (Mel-RoFormer becruily)...")
    failed_sep = download_separator_models()

    print("\n[2/2] Validando modelos RVC...")
    invalid_rvc = validate_existing_models(rvc_models_dir, RVC_EXPECTED_MIN_SIZES, RVC_DOWNLOAD_LINK)

    # Descargar modelos RVC si faltan
    rvc_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for model in rvc_model_names:
        model_path = rvc_models_dir / model
        if not model_path.exists():
            dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    all_invalid = failed_sep + invalid_rvc
    if all_invalid:
        print(f"\n[!] AVISO: {len(all_invalid)} archivo(s) no se pudieron validar: {', '.join(all_invalid)}")
        print("    La aplicación puede no funcionar correctamente. Revisa tu conexión a internet.")
    else:
        print("\n[+] Todos los modelos listos!")
        print("    Modelos del separador en: separator_models/")
        print("    Modelos RVC en: rvc_models/")
