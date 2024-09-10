import pandas as pd
import requests

def download_datasets(df, indices_metadata, output_file='data.csv'):
    """
    Download and save relevant datasets based on the FAISS search results.
    
    Args:
        df (pd.DataFrame): The dataframe containing the dataset metadata (including 'metadatasummary' and 'links').
        indices_metadata (list): The indices of the relevant datasets.
        output_file (str): The output CSV file to save the downloaded data.
    """
    relevant_datasets = df.iloc[indices_metadata]
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
