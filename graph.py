import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import re


def sanitize_uri_value(value: str) -> str:
    """Sanitize the input string to ensure it's a valid URI component."""
    sanitized_value = re.sub(r'[^\w]', '_', value)
    return sanitized_value


def preprocess_metadata(metadata_list):
    """
    Preprocess metadata to clean up unnecessary newlines, excessive spaces, and artifacts.
    :param metadata_list: List of metadata dictionaries.
    :return: Cleaned metadata list.
    """
    def clean_text(value):
        """Clean text by removing unwanted characters, newlines, and excessive spaces."""
        if not isinstance(value, str):
            return value
        # Replace newlines, carriage returns, and tabs with a single space
        cleaned_value = re.sub(r'[\r\n\t]+', ' ', value)
        # Remove multiple consecutive spaces
        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
        return cleaned_value.strip()  # Remove leading and trailing spaces

    cleaned_metadata_list = []
    for metadata in metadata_list:
        cleaned_metadata = {key: clean_text(value) for key, value in metadata.items()}
        cleaned_metadata_list.append(cleaned_metadata)

    return cleaned_metadata_list


def generate_dynamic_rdf_with_core(metadata_list, output_rdf_file='metadata_ontology.ttl'):
    """
    Generate RDF knowledge graph based on dataset metadata and save to file.
    :param metadata_list: List of metadata dictionaries for the relevant datasets.
    :param output_rdf_file: File path to save the RDF knowledge graph.
    """
    # Preprocess metadata to clean it up
    metadata_list = preprocess_metadata(metadata_list)

    g = Graph()

    # Define namespaces
    EX = Namespace("http://example.org/ontology/")
    SCHEMA = Namespace("https://schema.org/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("skos", SKOS)

    dataset_uris = []

    # Iterate through metadata and create RDF triples
    for idx, metadata in enumerate(metadata_list):
        # Create a unique URI for the dataset
        dataset_uri = URIRef(EX[f"Dataset_{idx + 1}"])
        dataset_uris.append(dataset_uri)
        g.add((dataset_uri, RDF.type, EX.Dataset))

        # Add metadata properties
        if 'title' in metadata and pd.notna(metadata['title']):
            g.add((dataset_uri, RDFS.label, Literal(metadata['title'])))

        if 'summary' in metadata and pd.notna(metadata['summary']):
            g.add((dataset_uri, EX.summary, Literal(metadata['summary'])))

        if 'publisher' in metadata and pd.notna(metadata['publisher']):
            g.add((dataset_uri, EX.publisher, Literal(metadata['publisher'])))

        if 'topic' in metadata and pd.notna(metadata['topic']):
            topics = [t.strip() for t in metadata['topic'].split(',')]
            for topic in topics:
                topic_uri = URIRef(EX[sanitize_uri_value(topic)])
                g.add((topic_uri, RDF.type, SKOS.Concept))
                g.add((topic_uri, SKOS.prefLabel, Literal(topic)))
                g.add((dataset_uri, EX.hasTopic, topic_uri))

        if 'links' in metadata and pd.notna(metadata['links']):
            links = [link.strip() for link in metadata['links'].split(',')]
            for link in links:
                g.add((dataset_uri, SCHEMA.url, URIRef(link)))

        if 'format' in metadata and pd.notna(metadata['format']):
            g.add((dataset_uri, EX.fileFormat, Literal(metadata['format'])))

    # Create links between datasets based on shared metadata
    create_dataset_level_links(g, dataset_uris, metadata_list, EX, SCHEMA)

    # Serialize the RDF graph
    g.serialize(output_rdf_file, format="turtle")
    print(f"Metadata RDF graph saved to {output_rdf_file}")


def create_dataset_level_links(graph, dataset_uris, metadata_list, EX, SCHEMA):
    """
    Create semantic links between datasets based on shared metadata values (e.g., shared topics or publishers).
    :param graph: RDF graph object.
    :param dataset_uris: List of dataset URIs.
    :param metadata_list: List of metadata dictionaries.
    :param EX: Example namespace.
    :param SCHEMA: Schema.org namespace.
    """
    for i, metadata_1 in enumerate(metadata_list):
        dataset_uri_1 = dataset_uris[i]
        for j, metadata_2 in enumerate(metadata_list):
            if i >= j:  # Avoid duplicate comparisons
                continue

            dataset_uri_2 = dataset_uris[j]

            # Check for shared topics
            if 'topic' in metadata_1 and 'topic' in metadata_2:
                topics_1 = set(metadata_1['topic'].split(','))
                topics_2 = set(metadata_2['topic'].split(','))
                common_topics = topics_1.intersection(topics_2)
                if common_topics:
                    graph.add((dataset_uri_1, SCHEMA.relatedTo, dataset_uri_2))

            # Check for shared publishers
            if (
                'publisher' in metadata_1 and 'publisher' in metadata_2 and
                metadata_1['publisher'] == metadata_2['publisher'] and
                pd.notna(metadata_1['publisher'])
            ):
                graph.add((dataset_uri_1, SCHEMA.relatedTo, dataset_uri_2))
