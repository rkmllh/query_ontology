from transformers import AutoTokenizer, AutoModel
import torch
from query_ontology.common.utils import (save_json, read_json)
import os
from safetensors.torch import save_file

class SentenceEmbedder:
    def __init__(self, model_name, device=None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            print(f"Using specified device: {device}")
            self.device = device
        
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, sentences, batch_size=8):

        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self.mean_pooling(outputs, inputs["attention_mask"])
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)
    

class OntologyEmbedder:
    def __init__(
        self,
        embedder,
        model_name,
        embeddings_path,
        ont_name="UBERON",
        batch_size=8,
    ):
        """
        Args:
            embedder: Sentence embedding model (must have .encode())
            model_name (str): Name of the embedding model
            embeddings_path (str): Base directory to save embeddings
            ont_name (str): Ontology prefix filter (default: UBERON)
            batch_size (int): Batch size for encoding
        """
        self.embedder = embedder
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.ont_name = ont_name
        self.batch_size = batch_size

    def get_embeddings(self, json_file):
        data = read_json(json_file)

        ontology_ids = []
        texts = []
        regions = []

        for region_name, entry in data.items():
            definitions = entry.get("definitions", {})
            for def_id, text in definitions.items():
                if def_id.startswith(self.ont_name):
                    ontology_ids.append(def_id)
                    texts.append(text)
                    regions.append(region_name)

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size
        )

        result = {}
        for oid, region, text, emb in zip(
            ontology_ids, regions, texts, embeddings
        ):
            result[oid] = {
                "region": region,
                "text": text,
                "embedding": emb
            }

        return result

    def save_embeddings(self, embeddings_dict):
        tensor_dict = {}
        metadata = {}

        for oid, info in embeddings_dict.items():
            tensor_dict[oid] = info["embedding"]

            metadata[oid] = {
                "region": info["region"],
                "text": info["text"]
            }

        safe_model_name = self.model_name.replace("/", "_")

        base, _ = os.path.splitext(self.embeddings_path)
        new_embeddings_dir = f"{base}_{safe_model_name}"

        os.makedirs(new_embeddings_dir, exist_ok=True)

        safetensor_path = os.path.join(
            new_embeddings_dir,
            f"{safe_model_name}.safetensors"
        )

        metadata_path = os.path.join(
            new_embeddings_dir,
            f"{safe_model_name}.json"
        )

        save_file(tensor_dict, safetensor_path)
        save_json(metadata, metadata_path)

        print("Saved to:", new_embeddings_dir)

    def process(self, json_file):
        embeddings = self.get_embeddings(json_file)
        self.save_embeddings(embeddings)
        return embeddings