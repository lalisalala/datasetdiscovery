import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import re

def sanitize_uri_value(value: str) -> str:
    """Sanitize the input string to ensure it's a valid URI component."""
    sanitized_value = re.sub(r'[^\w]', '_', value)
    return sanitized_value

def generate_dynamic_rdf_with_core(all_data_with_metadata, output_rdf_file='universal_data_ontology.ttl'):
    """Generate RDF including dataset structure (columns and data types) and save to file."""
    g = Graph()

    # Define namespaces
    EX = Namespace("http://example.org/ontology/")
    SCHEMA = Namespace("https://schema.org/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("skos", SKOS)

    all_rows = []
    dataset_uris = []

    # Iterate through each dataset and create RDF triples
    for idx, (metadata, df) in enumerate(all_data_with_metadata):
        # Create a unique URI for the dataset
        dataset_uri = URIRef(EX[f"Dataset_{str(idx + 1)}"])
        dataset_uris.append(dataset_uri)
        g.add((dataset_uri, RDF.type, URIRef(EX.Dataset)))

        # Add metadata as properties to the dataset
        for key, value in metadata.items():
            sanitized_key = sanitize_uri_value(key)
            property_uri = URIRef(EX[sanitized_key])
            g.add((dataset_uri, property_uri, Literal(value)))

        # Add column structure as part of the dataset RDF
        for column_name in df.columns:
            column_uri = URIRef(EX[sanitize_uri_value(column_name)])
            g.add((column_uri, RDF.type, EX.Column))
            g.add((column_uri, RDFS.label, Literal(column_name)))
            g.add((column_uri, EX.belongsToDataset, dataset_uri))

            # Add data type information
            dtype = str(df[column_name].dtype)
            g.add((column_uri, EX.dataType, Literal(dtype)))

        # Create triples for each row in the DataFrame
        for row_idx, row in df.iterrows():
            row_uri = URIRef(EX[f"Row_{str(idx + 1)}_{str(row_idx + 1)}"])
            g.add((row_uri, RDF.type, URIRef(EX.Row)))
            g.add((row_uri, URIRef(EX.partOf), dataset_uri))

            row_data = {'uri': row_uri, 'row': row, 'dataset_idx': idx, 'columns': df.columns}
            all_rows.append(row_data)

            # Add row data
            for column_name in df.columns:
                sanitized_column_name = sanitize_uri_value(column_name.strip())
                property_uri = URIRef(EX[sanitized_column_name])

                if pd.notna(row[column_name]):
                    g.add((row_uri, property_uri, Literal(row[column_name])))

            # Create category links if 'Category' exists
            if 'Category' in df.columns and pd.notna(row['Category']):
                category_label = row['Category']
                category_uri = URIRef(EX[sanitize_uri_value(category_label)])
                g.add((category_uri, RDF.type, SKOS.Concept))
                g.add((category_uri, SKOS.prefLabel, Literal(category_label)))
                g.add((row_uri, EX.hasCategory, category_uri))

    # Create dynamic row links based on shared values and high-level dataset links
    create_dynamic_links_between_datasets(g, all_rows, EX, SCHEMA)
    create_dataset_level_links(g, dataset_uris, SCHEMA)

    # Serialize the RDF graph
    g.serialize(output_rdf_file, format="turtle")
    print(f"RDF graph saved to {output_rdf_file}")

def create_dynamic_links_between_datasets(graph, all_rows, EX, SCHEMA):
    """Create semantic links based on shared values in rows across datasets."""
    for i, row_data_1 in enumerate(all_rows):
        row1_uri = row_data_1['uri']
        row1 = row_data_1['row']
        row1_columns = row_data_1['columns']

        for row_data_2 in all_rows[i + 1:]:
            row2_uri = row_data_2['uri']
            row2 = row_data_2['row']
            row2_columns = row_data_2['columns']

            # Find common columns between the two rows
            common_columns = set(row1_columns).intersection(set(row2_columns))
            identical = True

            for column in common_columns:
                if pd.notna(row1[column]) and pd.notna(row2[column]):
                    if row1[column] != row2[column]:
                        identical = False
                        # If some columns match, use schema:relatedTo
                        if column == 'Category':
                            graph.add((row1_uri, SCHEMA.relatedTo, row2_uri))

            # If all values in common columns are identical, use schema:sameAs
            if identical:
                graph.add((row1_uri, SCHEMA.sameAs, row2_uri))

def create_dataset_level_links(graph, dataset_uris, SCHEMA):
    """Create links between datasets based on shared metadata like topic."""
    for i, dataset_uri_1 in enumerate(dataset_uris):
        for dataset_uri_2 in dataset_uris[i + 1:]:
            graph.add((dataset_uri_1, SCHEMA.relatedTo, dataset_uri_2))
