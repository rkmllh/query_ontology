import os
import yaml

def init_ont(key: str = None):
    print("[!!] Initializing integration module...")
    local_bin = os.path.expanduser('~/.local/bin')
    if os.path.isdir(local_bin):
        os.environ['PATH'] += os.pathsep + local_bin

    os.environ['SCICRUNCH_API_KEY'] = key if key else "missing_key"

    print("[OK] Initialization complete.")
    
def load_params(path="config/parameters.yaml"):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)