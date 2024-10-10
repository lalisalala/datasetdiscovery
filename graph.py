import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF
import re

def sanitize_uri_value(value: str) -> str:
    """
    Sanitize the input string to ensure it's a valid URI component.
    Replaces spaces with underscores and removes invalid characters.
    """
    # Replace spaces with underscores and remove invalid characters
    sanitized_value = re.sub(r'[^\w]', '_', value)
    return sanitized_value

def generate_dynamic_rdf_with_core(csv_file, output_rdf_file='universal_data_ontology.ttl'):
    """
    Generate RDF from the provided CSV file, dynamically using the CSV columns as properties
    and treating each row as an individual resource.
    """
    # Create a new RDF graph
    g = Graph()

    # Define the namespace for your ontology
    EX = Namespace("http://example.org/ontology/")
    g.bind("ex", EX)

    # Load the CSV content into a DataFrame
    df = pd.read_csv(csv_file)

    # Define the class for a dataset
    Dataset = URIRef(EX.Dataset)
    hasMetadata = URIRef(EX.hasMetadata)

    # Add metadata for the dataset
    dataset_uri = URIRef(EX[f"Dataset_0"])
    g.add((dataset_uri, RDF.type, Dataset))
    g.add((dataset_uri, hasMetadata, Literal(f"Metadata for the dataset loaded from {csv_file}.")))

    # Iterate through each row in the CSV file and create triples dynamically
    for idx, row in df.iterrows():
        # Create a unique URI for each row (audit/record)
        row_uri = URIRef(EX[f"Row_{idx+1}"])

        # Add the row as an instance of ex:Row
        g.add((row_uri, RDF.type, URIRef(EX.Row)))

        # Link each row to the dataset
        g.add((row_uri, URIRef(EX.partOf), dataset_uri))

        # Iterate through each column in the row and add triples dynamically
        for column_name in df.columns:
            # Sanitize the column name to use it as a URI
            sanitized_column_name = sanitize_uri_value(column_name.strip())

            # Create a dynamic RDF property for this column
            property_uri = URIRef(EX[sanitized_column_name])

            # Add the property and value to the row
            g.add((row_uri, property_uri, Literal(row[column_name])))

    # Serialize the graph to a Turtle file
    g.serialize(output_rdf_file, format="turtle")
    print(f"RDF graph saved to {output_rdf_file}")
