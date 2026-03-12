# src/integration/query_ontology.py

import subprocess
import re
import ast
from nilearn import datasets
import os
from typing import Dict, Any, Tuple
import requests
from urllib.parse import quote
import csv
import json


BIOPORTAL_KEY = "6b0ad029-1de7-4fdd-a918-82a34dcf44e3"


def write_regions_from_atlas(atlas_name, filename):
    juelich = datasets.fetch_atlas_juelich(atlas_name, symmetric_split=False, resume=True, verbose=1)

    atlas_labels = juelich.labels
    dir_path = os.path.dirname(filename)
    
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    output_path = filename
    print(f"Writing atlas labels to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Atlas: {atlas_name}\n")
        for i, label in enumerate(atlas_labels[1:], start=1):  # Skip the first label
            line = f"{i}: {label}\n"
            f.write(line)
            print(line.strip())


def is_fma_or_uberon(ont_id: str, meta: Dict[str, Any]) -> bool:
    
    """
    Check whether a given ontology entry corresponds to FMA or UBERON.

    Parameters:
    - ont_id: str - the identifier of the ontology entry (e.g., 'FMA:68641').
    - meta: dict - metadata for the ontology entry, expected to possibly include 'iri'.

    Returns:
    - bool: True if the entry is from FMA or UBERON, False otherwise.
    """
    
    if isinstance(ont_id, str):
        if ont_id.startswith("FMA:") or ont_id.startswith("UBERON:"):
            return True

    iri = ""
    if isinstance(meta, dict):
        iri = meta.get("iri", "") or ""
    if isinstance(iri, str) and iri:
        iri_low = iri.lower()
        if "/fma" in iri_low or "fma" in iri_low and "ontology" in iri_low:
            return True
        if "uberon_" in iri_low or "obo/uberon" in iri_low:
            return True
    return False


def build_template_from_regions(
    regions_mapping: Dict[str, str],
    build_dict,
    t_value: int = 10,
    brain_dict: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    """
    Build a structured template of ontology definitions for brain regions.

    For each region label in `regions_mapping`:
    1. Uses the associated query to call `build_dict` and update `brain_dict`.
    2. Extracts ontology entries (FMA and UBERON only) for that query.
    3. For each ontology entry, selects a single string as the definition:
       - Prefers the first item in 'definitions' if available.
       - Falls back to the first 'label' if definitions are empty.
       - Falls back to the first 'synonym' if labels are empty.
       - Defaults to an empty string if none are available.
    4. Builds the output template in the format:
       {
           "region_label": {
               "query": <query string>,
               "definitions": {
                   "ONT_ID": <definition string>,
                   ...
               }
           },
           ...
       }

    Parameters:
    - regions_mapping: dict mapping atlas region labels to query strings used in build_dict.
      Example: {"GM Broca's area BA44": "BA44"}
    - build_dict: callable function that returns ontology entries for a given query and brain_dict.
    - t_value: int (default=10), optional parameter passed to build_dict.
    - brain_dict: dict, optional pre-existing dictionary of ontology entries; updated in-place.

    Returns:
    - template: dict mapping region labels to their query and filtered ontology definitions.
    - brain_dict: updated dictionary containing all ontology entries retrieved during processing.
    """
    
    if brain_dict is None:
        brain_dict = {}

    template: Dict[str, Any] = {}

    for region_label, query in regions_mapping.items():
        brain_dict = build_dict(brain_dict, query, t_value=t_value)

        ont_map = brain_dict.get(query)
        if ont_map is None:

            for k, v in brain_dict.items():
                if isinstance(k, str) and k.lower() == str(query).lower():
                    ont_map = v
                    break

        if not ont_map:
            template[region_label] = {"query": query, "definitions": {}}
            continue

        defs: Dict[str, str] = {}
        for ont_id, meta in ont_map.items():
            if not is_fma_or_uberon(str(ont_id), meta):
                continue  # skip non-FMA/UBERON entries

            definition_str = ""

            if isinstance(meta, dict):

                defs_list = meta.get("definitions")
                if isinstance(defs_list, list) and len(defs_list) > 0:
                    definition_str = defs_list[0].strip()
                elif isinstance(defs_list, str) and defs_list.strip():
                    definition_str = defs_list.strip()

                if not definition_str:
                    labels = meta.get("labels")
                    if isinstance(labels, list) and labels:
                        definition_str = labels[0].strip()

                if not definition_str:
                    syns = meta.get("synonyms")
                    if isinstance(syns, list) and syns:
                        definition_str = syns[0].strip()
            else:
                if isinstance(meta, str):
                    definition_str = meta.strip()

            defs[str(ont_id)] = definition_str

        template[region_label] = {"query": query, "definitions": defs}

    return template, brain_dict


def fetch_definition_from_bioportal(uberon_id):

    iri = f"http://purl.obolibrary.org/obo/{uberon_id.replace(':','_')}"
    encoded_iri = quote(iri, safe='')
    url = f"https://data.bioontology.org/ontologies/UBERON/classes/{encoded_iri}"
    headers = {"Authorization": f"apikey token={BIOPORTAL_KEY}"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            defs = data.get("definition", [])
            if not defs:
                defs = data.get("comment", [])
            return defs
        else:
            print(f"[BioPortal] Error {resp.status_code} for {uberon_id}")
    except Exception as e:
        print(f"[BioPortal] Exception for {uberon_id}: {e}")
    return []


def build_dict(brain_dict, term, t_value=10):

    result = subprocess.run(
        ["scig", "-t", str(t_value), "t", term, "-v"],
        capture_output=True,
        text=True
    )
    
    output_str = result.stdout
    lines = output_str.splitlines()
    if not lines:
        return brain_dict

    zone = lines[0].strip()
    brain_dict[zone] = {}

    kv_pattern = re.compile(r'\s*(\w+):\s*(.*)')
    current_ontology = None

    for line in lines[1:]:
        line = line.rstrip()
        if not line.strip():
            continue

        if re.match(r'^\s*[A-Z]+[:_]\d+', line):
            current_ontology = line.strip()
            if current_ontology not in brain_dict[zone]:
                brain_dict[zone][current_ontology] = {}
            continue

        if current_ontology is not None:
            match = kv_pattern.match(line)
            if match:
                key, value = match.groups()
                if value.startswith("[") and value.endswith("]"):
                    try:
                        value = ast.literal_eval(value)
                    except:
                        pass
                if value in ["True", "False"]:
                    value = (value == "True")
                brain_dict[zone][current_ontology][key] = value

    for ont in brain_dict[zone]:
        if ont.startswith("UBERON:"):
            defs = fetch_definition_from_bioportal(ont)
            brain_dict[zone][ont]['definitions'] = defs

    return brain_dict


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


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=5, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)