import click
from query_ontology.query_engine.query_ontology import BrainDictBuilder
from query_ontology.common.utils import load_params
from query_ontology.common.utils import init_ont
from query_ontology.query_engine.SentenceEmbedder import SentenceEmbedder, OntologyEmbedder


@click.group()
def cli():
    pass


@cli.command()
@click.option("--t_value", default=10, show_default=True, type=int, help="Threshold value")
def create_dict(t_value):
    BrainDictBuilder(t_value=t_value)


@cli.command()
@click.option("--t_value", default=10, show_default=True, type=int, help="Threshold value")
def build_pipeline(t_value):
    """Create solved-region definitions and then generate ontology embeddings."""
    print("[1/2] Building solved regions...")
    builder = BrainDictBuilder(t_value=t_value)

    print("[2/2] Generating ontology embeddings...")
    config = load_params(path="config/parameters.yaml")
    json_regions = config["json_regions"]
    embeddings_path = config["embeddings_path"]
    model_name = config["model_name"]

    embedder = SentenceEmbedder(model_name, device="cuda:1")
    ontology_embedder = OntologyEmbedder(
        embedder=embedder,
        model_name=model_name,
        embeddings_path=embeddings_path,
        ont_name="UBERON",
        batch_size=16,
    )
    ontology_embedder.process(json_regions)


@cli.command()
def create_embedding():
    
    config = load_params(path="config/parameters.yaml")
    json_regions = config["json_regions"]
    embeddings_path = config["embeddings_path"]
    model_name = config["model_name"]

    embedder = SentenceEmbedder(model_name, device="cuda:1")

    ontology_embedder = OntologyEmbedder(
        embedder=embedder,
        model_name=model_name,
        embeddings_path=embeddings_path,
        ont_name="UBERON",
        batch_size=16
    )

    ontology_embedder.process(json_regions)


if __name__ == "__main__":
    cli()