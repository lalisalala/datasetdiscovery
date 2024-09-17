import pandas as pd
import requests
import yaml

def load_query_from_yaml(yaml_path='config/query.yaml'):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Debugging: print out the query
    query = config['query']
    print(f"Loaded query from YAML: {query}")  # This will print the query to the console
    return query


import os
import pandas as pd
import requests

def download_datasets(relevant_datasets, output_file='data.csv'):
    """
    Download and save relevant datasets based on the first FAISS or LLM selection.
    Include the LLM-generated metadata summaries as headers, and always create a new 'data.csv' file.
    """
    all_data_with_metadata = []

    for _, row in relevant_datasets.iterrows():
        dataset_link = row['links']
        metadata_summary = row['metadatasummary']

        try:
            response = requests.get(dataset_link)
            response.raise_for_status()

            # Handle different file formats
            if dataset_link.endswith('.csv'):
                temp_df = pd.read_csv(dataset_link)
            elif dataset_link.endswith('.json'):
                temp_df = pd.read_json(dataset_link)
            elif dataset_link.endswith('.xlsx'):
                temp_df = pd.read_excel(dataset_link)
            else:
                print(f"Unsupported file format: {dataset_link}")
                continue

            # Append the metadata summary and dataframe to the list
            all_data_with_metadata.append((metadata_summary, temp_df))

        except Exception as e:
            print(f"Failed to download dataset from {dataset_link}: {e}")

    # After gathering all datasets, save them
    save_data_with_llm_metadata_header(all_data_with_metadata, output_file)

def save_data_with_llm_metadata_header(all_data, output_file):
    """
    Save downloaded datasets with LLM-generated metadata summaries as headers.
    Always overwrite the file and ensure each dataset's content is saved.
    """
    # Open file for writing to overwrite it if it exists
    with open(output_file, 'w') as f:
        for metadata_summary, df in all_data:
            # Write the metadata summary as a header (comment line)
            f.write(f"# {metadata_summary}\n")
            
            # Append dataset content to the file, excluding index
            df.to_csv(f, index=False)
            
            # Add separation between datasets for readability
            f.write("\n\n")
    
    print(f"Downloaded datasets saved to {output_file}.")

