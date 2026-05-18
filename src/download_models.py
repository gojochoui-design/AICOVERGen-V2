from pathlib import Path
import os
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'

mdxnet_models_dir.mkdir(parents=True, exist_ok=True)
rvc_models_dir.mkdir(parents=True, exist_ok=True)

# Known minimum sizes for MDX models (in bytes).
# These are approximate lower bounds; a real, complete model should always be larger.
# If a downloaded file is smaller, it is almost certainly truncated/corrupted.
MDX_EXPECTED_MIN_SIZES = {
    'UVR-MDX-NET-Voc_FT.onnx':    45_000_000,   # ~50-80MB
    'UVR-MDX-NET-Inst_HQ_4.onnx': 40_000_000,   # ~45-70MB
    'UVR_MDXNET_KARA_2.onnx':     40_000_000,   # ~45-70MB
    'Reverb_HQ_By_FoxJoy.onnx':   40_000_000,   # ~45-70MB
}

RVC_EXPECTED_MIN_SIZES = {
    'hubert_base.pt':  180_000_000,  # ~190MB
    'rmvpe.pt':         50_000_000,  # ~55MB
}

MAX_RETRIES = 3


def dl_model(link, model_name, dir_name, force=False):
    """Download a model file with size validation and automatic retry.
    
    Args:
        link: Base URL for downloading
        model_name: Filename of the model
        dir_name: Directory to save the model (Path object)
        force: If True, re-download even if file exists (used for corrupted files)
    
    Returns:
        bool: True if the model is valid and ready to use, False otherwise
    """
    model_path = dir_name / model_name
    
    # Determine expected minimum size
    expected_min = MDX_EXPECTED_MIN_SIZES.get(model_name, RVC_EXPECTED_MIN_SIZES.get(model_name, 0))
    
    # Check if existing file is valid
    if not force and model_path.exists():
        if expected_min > 0:
            actual_size = os.path.getsize(model_path)
            if actual_size >= expected_min:
                # File exists and is large enough, skip download
                return True
            else:
                # File exists but is too small (corrupted/truncated), re-download
                print(f"[!] {model_name} exists but is too small ({actual_size / 1024 / 1024:.1f}MB < {expected_min / 1024 / 1024:.1f}MB expected). Re-downloading...")
                os.remove(model_path)
        else:
            # No size info available, trust existing file
            return True
    
    # If force=True and file exists, remove it
    if force and model_path.exists():
        os.remove(model_path)
    
    # Download with retries
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {model_name}... (attempt {attempt}/{MAX_RETRIES})")
            with requests.get(f'{link}{model_name}', stream=True, timeout=120) as r:
                r.raise_for_status()
                # Get expected total size from headers if available
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                
                # Validate downloaded file size
                actual_size = os.path.getsize(model_path)
                
                # Check against Content-Length header
                if total_size > 0 and actual_size != total_size:
                    print(f"[!] {model_name}: downloaded {actual_size} bytes but expected {total_size} bytes from server.")
                    if attempt < MAX_RETRIES:
                        print(f"    Retrying...")
                        os.remove(model_path)
                        continue
                    else:
                        print(f"    Max retries reached. File may be incomplete.")
                        return False
                
                # Check against known minimum size
                if expected_min > 0 and actual_size < expected_min:
                    print(f"[!] {model_name}: downloaded file is {actual_size / 1024 / 1024:.1f}MB but expected at least {expected_min / 1024 / 1024:.1f}MB.")
                    if attempt < MAX_RETRIES:
                        print(f"    Retrying...")
                        os.remove(model_path)
                        continue
                    else:
                        print(f"    Max retries reached. File may be corrupted.")
                        return False
                
                print(f"[+] {model_name} downloaded successfully ({actual_size / 1024 / 1024:.1f}MB)")
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"[!] Error downloading {model_name} (attempt {attempt}/{MAX_RETRIES}): {e}")
            if model_path.exists():
                os.remove(model_path)
            if attempt < MAX_RETRIES:
                print(f"    Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                print(f"    Max retries reached. Download failed.")
                return False
    
    return False


def validate_existing_models(dir_name, expected_sizes, link):
    """Validate all existing model files in a directory and re-download any corrupted ones.
    
    Args:
        dir_name: Directory containing models (Path object)
        expected_sizes: Dict mapping filename to minimum expected size in bytes
        link: Download URL base for re-downloading
    
    Returns:
        list: Names of models that are invalid and could not be fixed
    """
    invalid = []
    for model_name, min_size in expected_sizes.items():
        model_path = dir_name / model_name
        if model_path.exists():
            actual_size = os.path.getsize(model_path)
            if actual_size < min_size:
                print(f"[!] {model_name} is corrupted ({actual_size / 1024 / 1024:.1f}MB < {min_size / 1024 / 1024:.1f}MB expected). Re-downloading...")
                os.remove(model_path)
                if not dl_model(link, model_name, dir_name):
                    invalid.append(model_name)
        # If model doesn't exist, skip (will be downloaded in main flow)
    
    return invalid


if __name__ == '__main__':
    print("=" * 60)
    print("AICoverGen Remaster - Model Download & Validation")
    print("=" * 60)
    
    # First validate any existing models
    print("\n[1/3] Validating existing MDX models...")
    invalid_mdx = validate_existing_models(mdxnet_models_dir, MDX_EXPECTED_MIN_SIZES, MDX_DOWNLOAD_LINK)
    
    print("\n[2/3] Validating existing RVC models...")
    invalid_rvc = validate_existing_models(rvc_models_dir, RVC_EXPECTED_MIN_SIZES, RVC_DOWNLOAD_LINK)
    
    # Download any missing models
    print("\n[3/3] Downloading missing models...")
    mdx_model_names = ['UVR-MDX-NET-Inst_HQ_4.onnx', 'UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        model_path = mdxnet_models_dir / model
        if not model_path.exists():
            dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        model_path = rvc_models_dir / model
        if not model_path.exists():
            dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    # Final summary
    all_invalid = invalid_mdx + invalid_rvc
    if all_invalid:
        print(f"\n[!] WARNING: {len(all_invalid)} model(s) could not be validated: {', '.join(all_invalid)}")
        print("    The application may not work correctly. Check your internet connection and try again.")
    else:
        print('\n[+] All models ready!')
