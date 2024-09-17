import pandas as pd
import requests
import os 
import yaml

def load_query_from_yaml(yaml_path='config/query.yaml'):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Debugging: print out the query
    query = config['query']
    print(f"Loaded query from YAML: {query}")  # This will print the query to the console
    return query


def download_datasets(relevant_datasets, output_file='data.csv'):
    """
    Download and save relevant datasets based on the first FAISS or LLM selection.
    Include the LLM-generated metadata summaries as headers.
    """
    all_data_with_metadata = []

    for _, row in relevant_datasets.iterrows():
        dataset_link = row['links']
        metadata_summary = row['metadatasummary']

        try:
            response = requests.get(dataset_link)
            response.raise_for_status()

            if dataset_link.endswith('.csv'):
                temp_df = pd.read_csv(dataset_link)
            elif dataset_link.endswith('.json'):
                temp_df = pd.read_json(dataset_link)
            elif dataset_link.endswith('.xlsx'):
                temp_df = pd.read_excel(dataset_link)
            else:
                print(f"Unsupported file format: {dataset_link}")
                continue

            all_data_with_metadata.append((metadata_summary, temp_df))
        except Exception as e:
            print(f"Failed to download dataset from {dataset_link}: {e}")

    save_data_with_llm_metadata_header(all_data_with_metadata, output_file)

def save_data_with_llm_metadata_header(all_data, output_file):
    """
    Save downloaded datasets with LLM-generated metadata summaries as headers.
    """
    with open(output_file, 'w') as f:
        for metadata_summary, df in all_data:
            # Remove '#' to ensure the metadata is not treated as a comment
            f.write(f"{metadata_summary}\n")  # No '#' prefix here
            df.to_csv(f, index=False)
            f.write("\n\n")

