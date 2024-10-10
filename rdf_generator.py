#rdf_generation.py
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import pandas as pd
import re

def generate_dynamic_rdf_from_datasets(datasets: pd.DataFrame, output_rdf_file='data_ontology.ttl'):
    """
    Generate an RDF graph that stores both dataset metadata and the actual content of the datasets.
    
    Args:
        datasets (pd.DataFrame): The DataFrame containing metadata and links to the datasets.
        output_rdf_file (str): The RDF file to save.
    """
    g = Graph()
    EX = Namespace("http://example.org/ontology/")
    g.bind("ex", EX)

    # Helper function to sanitize dataset titles for use as URIs
    def sanitize_for_uri(text):
        return re.sub(r'\W+', '_', text)  # Replace non-alphanumeric characters with underscores

    # Iterate over each dataset
    for _, row in datasets.iterrows():
        # Use sanitized dataset title as a unique identifier for each dataset
        dataset_title = sanitize_for_uri(row['title'])
        dataset_uri = URIRef(EX[f"Dataset_{dataset_title}"])  # Unique identifier based on title
        
        # Add dataset metadata (e.g., title, link, etc.)
        g.add((dataset_uri, RDF.type, EX.Dataset))
        g.add((dataset_uri, EX.hasTitle, Literal(row['title'])))
        g.add((dataset_uri, EX.hasLink, Literal(row['links'])))
        
        # Download the dataset and parse its content
        dataset_content = download_and_extract_dataset(row['links'])
        
        # Add the dataset content (rows and columns) to the RDF graph
        if dataset_content is not None:
            for i, content_row in dataset_content.iterrows():
                row_uri = URIRef(EX[f"Row_{dataset_title}_{i}"])  # Unique URI for each row
                
                g.add((dataset_uri, EX.hasRow, row_uri))  # Link each row to the dataset
                for col in dataset_content.columns:
                    column_value = content_row[col]
                    sanitized_col = sanitize_for_uri(col)
                    g.add((row_uri, EX[sanitized_col], Literal(column_value)))  # Add column data as RDF triple

    # Serialize the RDF graph to Turtle format
    g.serialize(output_rdf_file, format="turtle")
    print(f"RDF graph saved to {output_rdf_file}")
    
def download_and_extract_dataset(link):
    """
    Download the dataset and return it as a DataFrame. This function can be extended to handle multiple formats.
    """
    try:
        # Example for CSV datasets
        df = pd.read_csv(link)
        return df
    except Exception as e:
        print(f"Error downloading dataset from {link}: {e}")
        return None