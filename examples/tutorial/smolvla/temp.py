from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

folder = Path(snapshot_download("lerobot/smolvla_base"))

for p in folder.rglob("*.safetensors"):
    try:
        stats = load_file(p)
    except Exception:
        continue

    print("\n====", p.name, "====")
    for k in stats.keys():
        if "state" in k or "action" in k or "observation" in k:
            print(k, tuple(stats[k].shape))