import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF
import re

def sanitize_uri_value(value: str) -> str:
    """
    Sanitize the input string to ensure it's a valid URI component.
    Replaces spaces with underscores and removes invalid characters.
    """
    sanitized_value = re.sub(r'[^\w]', '_', value)
    return sanitized_value

def generate_dynamic_rdf_with_core(all_data_with_metadata, output_rdf_file='universal_data_ontology.ttl'):
    """
    Generate RDF from the provided metadata and datasets.
    Now uses a list of tuples, where each tuple contains metadata and a DataFrame for the dataset.
    """
    # Create a new RDF graph
    g = Graph()

    # Define the namespace for your ontology
    EX = Namespace("http://example.org/ontology/")
    g.bind("ex", EX)

    # Define a class for the dataset
    Dataset = URIRef(EX.Dataset)

    # Iterate through each dataset and create triples dynamically
    for idx, (metadata, df) in enumerate(all_data_with_metadata):
        # Create a unique URI for the dataset
        dataset_uri = URIRef(EX[f"Dataset_{str(idx + 1)}"])
        g.add((dataset_uri, RDF.type, Dataset))

        # Add metadata as properties to the dataset
        for key, value in metadata.items():
            # Sanitize the metadata key and create a dynamic RDF property
            sanitized_key = sanitize_uri_value(key)
            property_uri = URIRef(EX[sanitized_key])
            g.add((dataset_uri, property_uri, Literal(value)))

        # Iterate through each row in the DataFrame and create triples
        for row_idx, row in df.iterrows():
            # Create a unique URI for each row (audit/record)
            row_uri = URIRef(EX[f"Row_{str(idx + 1)}_{str(row_idx + 1)}"])

            # Add the row as an instance of ex:Row and link it to the dataset
            g.add((row_uri, RDF.type, URIRef(EX.Row)))
            g.add((row_uri, URIRef(EX.partOf), dataset_uri))

            # Dynamically generate properties based on the column names
            for column_name in df.columns:
                # Sanitize the column name to create a valid URI
                sanitized_column_name = sanitize_uri_value(column_name.strip())

                # Create a dynamic RDF property for this column
                property_uri = URIRef(EX[sanitized_column_name])

                # Add the property and value to the row (skipping NaNs)
                if pd.notna(row[column_name]):
                    g.add((row_uri, property_uri, Literal(row[column_name])))

    # Serialize the graph to a Turtle file
    g.serialize(output_rdf_file, format="turtle")
    print(f"RDF graph saved to {output_rdf_file}")
