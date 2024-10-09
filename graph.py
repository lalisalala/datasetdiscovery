import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
import re

def sanitize_uri_value(value: str) -> str:
    """
    Sanitize the input string to ensure it's a valid URI component.
    Replaces spaces with underscores and removes invalid characters.
    """
    return value.replace(" ", "_").replace("/", "_").replace("?", "").replace("&", "").replace(",", "")

def generate_dynamic_rdf_with_core(csv_file, output_rdf_file='universal_data_ontology.ttl'):
    """
    Generate RDF from the provided data.csv, considering that metadata is a header
    and associating it with each dataset.
    """
    # Create a new RDF graph
    g = Graph()

    # Define the namespace for your core ontology
    EX = Namespace("http://example.org/ontology/")
    g.bind("ex", EX)

    # Core RDF Classes
    Dataset = URIRef(EX.Dataset)
    Field = URIRef(EX.Field)
    Value = URIRef(EX.Value)

    # Core RDF Properties
    hasField = URIRef(EX.hasField)
    hasValue = URIRef(EX.hasValue)
    hasTitle = URIRef(EX.hasTitle)
    hasLink = URIRef(EX.hasLink)
    hasDescription = URIRef(EX.hasDescription)
    hasMetadata = URIRef(EX.hasMetadata)  # New property to capture metadata

    # Load the CSV content
    with open(csv_file, 'r') as file:
        lines = file.readlines()

    current_dataset_metadata = None
    all_data_with_metadata = []
    dataset_started = False
    current_data = []

    # Iterate through the lines and dynamically generate RDF triples
    for line in lines:
        line = line.strip()

        # Detect new dataset metadata header
        if line.startswith("Dataset Metadata:"):
            if dataset_started:
                # Process the previous dataset before starting a new one
                all_data_with_metadata.append((current_dataset_metadata, current_data))
                current_data = []  # Reset the dataset content

            # Capture the metadata for the new dataset
            current_dataset_metadata = re.sub(r"Dataset Metadata:\s*", "", line)
            dataset_started = True

        # Detect CSV data (i.e., after the metadata header)
        elif dataset_started and "," in line:
            current_data.append(line)

    # Process the last dataset
    if current_data:
        all_data_with_metadata.append((current_dataset_metadata, current_data))

    # Process each dataset with its metadata
    for idx, (metadata, data_lines) in enumerate(all_data_with_metadata):
        # Create a unique URI for the dataset
        dataset_uri = URIRef(EX[sanitize_uri_value(f"Dataset_{idx}")])

        # Add the dataset as a subject in RDF
        g.add((dataset_uri, RDF.type, Dataset))

        # Add the metadata as a property
        g.add((dataset_uri, hasMetadata, Literal(metadata)))

        # Process the dataset CSV data (starting from the second line, which contains actual data)
        if data_lines:
            header = data_lines[0].split(",")  # Extract the CSV header (Category, AuditTitle, OutlineScope)
            for line in data_lines[1:]:
                row = line.split(",")

                # Add fields and values based on the CSV header
                for col_name, value in zip(header, row):
                    field_uri = URIRef(EX[sanitize_uri_value(f"has{col_name.strip()}")])  # Sanitize field URI
                    g.add((dataset_uri, field_uri, Literal(value)))  # Add field and value triples

    # Serialize the graph to an RDF file (Turtle format)
    g.serialize(output_rdf_file, format="turtle")
    print(f"RDF graph saved to {output_rdf_file}")
