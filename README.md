# Brain Tumor Detection

This project builds a brain-region ontology dictionary from anatomical queries and generates UBERON embeddings for those definitions.

It is organized around a small CLI that:
- queries anatomical/ontology information using the `scig` tool
- filters the results to FMA and UBERON entries
- saves a solved region dictionary as JSON/CSV
- embeds the collected UBERON textual definitions using a Hugging Face sentence model

## Project structure

- `query_ontology/__main__.py` — package entry point
- `query_ontology/main.py` — command-line interface
- `query_ontology/query_engine/query_ontology.py` — ontology query and dictionary-building logic
- `query_ontology/query_engine/SentenceEmbedder.py` — embedding generation logic
- `query_ontology/common/utils.py` — JSON/CSV helpers and config loading
- `config/parameters.yaml` — runtime settings and output paths
- `outs/` — generated region and embedding outputs
- `run_pipeline.sh` — convenience script for the full workflow

## Requirements

- Python 3.12+
- project virtual environment
- `scig` available in PATH
- internet access for ontology lookup and model downloads

Install dependencies in your environment:

```bash
python -m pip install -r requirements.txt
```

If you are using the local project virtual environment:

```bash
source venv/bin/activate
```

## Configuration

The main configuration file is:

- `config/parameters.yaml`

Current defaults:

```yaml
atlas_name: maxprob-thr0-1mm
output_file: integration/regions_labels.txt

json_regions: outs/solved_regions.json
csv_regions: outs/solved_regions.csv
embeddings_path: outs/uberon_embeddings.json
regions_query: integration/regions_query.json
model_name: NeuML/pubmedbert-base-embeddings
```

These paths control where the generated region metadata and embedding artifacts are saved.

## Usage

### 1) Build the solved-region dictionary

```bash
python -m query_ontology create-dict
```

This creates the ontology mapping and writes:
- `outs/solved_regions.json`
- `outs/solved_regions.csv`

### 2) Generate embeddings

```bash
python -m query_ontology create-embedding
```

This reads the solved regions JSON, filters UBERON entries, encodes their definitions with the configured model, and saves embedding artifacts to the output directory associated with `outs/uberon_embeddings.json`.

### 3) Run the full workflow in sequence

```bash
bash run_pipeline.sh
```

This executes:

```bash
python -m query_ontology create-dict
python -m query_ontology create-embedding
```

## Output files

The pipeline produces files in `outs/`:

- `outs/solved_regions.json` — structured region definitions and ontology metadata
- `outs/solved_regions.csv` — tabular export of the same data
- embedding files under the directory derived from `outs/uberon_embeddings.json` (for example `.safetensors` and `.json` metadata files)

## Notes

- `create-dict` is the data-construction phase.
- `create-embedding` is the vectorization phase.
- `run_pipeline.sh` is the simplest way to run the full workflow end-to-end.
- The project currently uses a CUDA device for embedding generation (`cuda:1` in `query_ontology/main.py`), so GPU availability may matter when running the embedding step.

## Troubleshooting

If the CLI fails:
- verify the Python environment is activated
- verify `scig` is installed and available in PATH
- verify your ontology/API keys are configured in `config/keys.yaml`
- confirm the output directories are writable