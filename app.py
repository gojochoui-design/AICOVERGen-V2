import os
import sys

# Download and validate models before launching the WebUI
print("=" * 50)
print("AICoverGen Remaster - Starting up...")
print("=" * 50)

# Run model download with validation
download_result = os.system("python src/download_models.py")
if download_result != 0:
    print("[!] Warning: Model download exited with errors. Some models may be missing.")
    print("    You can retry manually with: python src/download_models.py")

args = " ".join(sys.argv[1:])
cmd = f"python src/webui.py {args}"

os.system(cmd)
