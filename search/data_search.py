import pandas as pd
import requests


def download_datasets(relevant_datasets, output_file='data.csv', successful_links=None):
    """
    Download and save relevant datasets based on the first FAISS or LLM selection.
    Include the LLM-generated metadata summaries as headers.
    Track and return successful dataset links for logging.
    
    Args:
        relevant_datasets (pd.DataFrame): The dataframe of relevant datasets.
        output_file (str): The file to save the downloaded datasets.
        successful_links (list): A list to track successfully downloaded dataset links.
    """
    if successful_links is None:
        successful_links = []

    all_data_with_metadata = []

    for _, row in relevant_datasets.iterrows():
        dataset_link = row['links']
        metadata_summary = row['metadatasummary']

        try:
            # Attempt to download the dataset
            response = requests.get(dataset_link)
            response.raise_for_status()

            # Check the file format and read the dataset accordingly
            if dataset_link.endswith('.csv'):
                temp_df = pd.read_csv(dataset_link)
            elif dataset_link.endswith('.json'):
                temp_df = pd.read_json(dataset_link)
            elif dataset_link.endswith('.xlsx'):
                temp_df = pd.read_excel(dataset_link)
            else:
                print(f"Unsupported file format: {dataset_link}")
                continue

            # Append the metadata and dataset to the list
            all_data_with_metadata.append((metadata_summary, temp_df))

            # Track the successful link
            successful_links.append(dataset_link)

        except Exception as e:
            print(f"Failed to download dataset from {dataset_link}: {e}")

    # Save the datasets with LLM metadata headers
    save_data_with_llm_metadata_header(all_data_with_metadata, output_file)

    # Return the list of successful links for logging
    return successful_links



def save_data_with_llm_metadata_header(all_data, output_file):
    """
    Save downloaded datasets with LLM-generated metadata summaries as headers.

    Args:
        all_data (list): A list of tuples where each tuple contains a metadata summary and a dataset (DataFrame).
        output_file (str): The output file to save the datasets.
    """
    with open(output_file, 'w') as f:
        for metadata_summary, df in all_data:
            # Write the metadata summary as a header (no '#' prefix)
            f.write(f"Dataset Metadata: {metadata_summary}\n")

            # Write the dataset to the file without index
            df.to_csv(f, index=False)

            # Add some space between datasets for readability
            f.write("\n\n")
