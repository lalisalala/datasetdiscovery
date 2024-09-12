import pandas as pd
import requests

def download_datasets(relevant_datasets, output_file='data.csv'):
    """
    Download and save relevant datasets based on the FAISS search results or LLM selection.

    Args:
        relevant_datasets (pd.DataFrame): A dataframe containing the relevant dataset metadata, including 'metadatasummary' and 'links'.
        output_file (str): The output CSV file to save the downloaded data.
    """
    all_data_with_metadata = []
    
    for _, row in relevant_datasets.iterrows():
        dataset_link = row['links']
        metadata_summary = row['metadatasummary']  # Extract the metadata summary
        
        print(f"Attempting to download dataset from {dataset_link}")
        
        try:
            response = requests.get(dataset_link)
            response.raise_for_status()

            # Handle different file formats
            if dataset_link.endswith('.csv'):
                temp_df = pd.read_csv(dataset_link)
            elif dataset_link.endswith('.json'):
                temp_df = pd.read_json(dataset_link)
            elif dataset_link.endswith('.xlsx'):
                temp_df = pd.read_excel(dataset_link)  # Excel file support
            else:
                print(f"Unsupported file format: {dataset_link}")
                continue
            
            print(f"Successfully downloaded dataset from {dataset_link}")
            print(f"Sample Data from dataset before preprocessing:\n{temp_df.head()}")
            
            all_data_with_metadata.append((metadata_summary, temp_df))  # Append metadata and dataset as a tuple
        except Exception as e:
            print(f"Failed to download dataset from {dataset_link}: {e}")
    
    if all_data_with_metadata:
        save_data_with_llm_metadata_header(all_data_with_metadata, output_file)
    else:
        print("No datasets to download.")

def save_data_with_llm_metadata_header(all_data, output_file):
    """
    Save downloaded data along with LLM-generated metadata summaries as a header.

    Args:
        all_data (list of tuples): List of tuples where each tuple contains a metadata summary and the corresponding dataset.
        output_file (str): The output CSV file to save the data.
    """
    with open(output_file, 'w') as f:
        for metadata_summary, df in all_data:
            # Write the metadata summary as a header
            f.write(f"# {metadata_summary}\n")
            df.to_csv(f, index=False)
            f.write("\n\n")  # Add space between datasets

    print(f"Relevant datasets saved to {output_file} with metadata headers.")
