import pandas as pd
import requests

def download_datasets(relevant_datasets, output_file='data.csv'):
    """
    Download and save relevant datasets based on the FAISS search results or LLM selection.

    Args:
        relevant_datasets (pd.DataFrame): A dataframe containing the relevant dataset metadata, including 'links'.
        output_file (str): The output CSV file to save the downloaded data.
    """
    all_data = []
    
    for _, row in relevant_datasets.iterrows():
        dataset_link = row['links']
        
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
            
            all_data.append(temp_df)
        except Exception as e:
            print(f"Failed to download dataset from {dataset_link}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined DataFrame before saving:\n{combined_df.head()}")
        
        combined_df.to_csv(output_file, index=False)
        print(f"Relevant datasets saved to {output_file}")
    else:
        print("No datasets to download.")

def save_data_with_llm_metadata_header(relevant_datasets, combined_df, output_file='data.csv'):
    """
    Save the dataset to a CSV file, including the generated LLM metadata summary as a header.
    
    Args:
        relevant_datasets (pd.DataFrame): DataFrame containing the relevant datasets and their metadata summaries.
        combined_df (pd.DataFrame): The DataFrame containing the actual dataset content that was downloaded.
        output_file (str): The path of the output CSV file.
    """
    # Extract the LLM-generated summary for the relevant datasets
    metadata_summary = '\n\n'.join(relevant_datasets['metadatasummary'].tolist())

    # Open the output file and write the metadata summary at the top
    with open(output_file, 'w') as f:
        # Write the metadata summary as a header
        f.write(f"LLM-Generated Metadata Summary:\n{metadata_summary}\n\n")

    # Append the actual dataset content
    combined_df.to_csv(output_file, mode='a', index=False)  # Append mode
    print(f"Data saved to {output_file} with LLM-generated metadata as a header.")