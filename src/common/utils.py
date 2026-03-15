import csv
import json
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


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=5, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_csv(data, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "query", "ontology_id", "definition"])

        for label, content in data.items():
            query = content.get("query", "")
            definitions = content.get("definitions", {})

            for ont_id, definition in definitions.items():
                writer.writerow([
                    label,
                    query,
                    ont_id,
                    definition
                ])