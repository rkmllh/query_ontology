import click
from src.integration.query_ontology import build_dict, build_template_from_regions
from src.common.utils import load_params, save_json, json_to_csv, read_json
from src.common.utils import init_ont
from src.integration.SentenceEmbedder import SentenceEmbedder, OntologyEmbedder


@click.group()
def cli():
    pass


@cli.command()
@click.option("--t_value", default=10, show_default=True, type=int, help="Threshold value")
def create_dict(t_value):

    keys = load_params(path="config/keys.yaml")
    init_ont(key=keys["ont_key"])

    parameters = load_params(path="config/parameters.yaml")
    json_regions = parameters["json_regions"]
    csv_regions = parameters["csv_regions"]
    regions_query = read_json(parameters["regions_query"])

    template, _ = build_template_from_regions(
        regions_query,
        build_dict,
        t_value=t_value
    )

    save_json(template, json_regions)
    json_to_csv(template, csv_regions)

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