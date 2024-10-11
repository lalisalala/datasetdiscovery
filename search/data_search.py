import pandas as pd
import requests


import pandas as pd
import requests

def download_datasets(relevant_datasets, output_file='data.csv', successful_links=None):
    """
    Download and save relevant datasets based on the first FAISS or LLM selection.
    Include the LLM-generated metadata alongside the dataset in the CSV.
    Track and return successful dataset links for logging.
    
    Args:
        relevant_datasets (pd.DataFrame): The dataframe of relevant datasets.
        output_file (str): The file to save the downloaded datasets.
        successful_links (list): A list to track successfully downloaded dataset links.
    """
    if successful_links is None:
        successful_links = []

    all_data_with_metadata = []

    with open(output_file, 'w') as f:
        for _, row in relevant_datasets.iterrows():
            dataset_link = row['links']
            metadata = {
                "title": row.get("title", ""),
                "summary": row.get("summary", ""),
                "links": dataset_link,
                "name": row.get("name", "")
            }

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

                # Write metadata information to the output CSV file
                f.write("Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Write the dataset to the output CSV file
                temp_df.to_csv(f, index=False)
                
                # Add space between different datasets for readability
                f.write("\n\n")

                # Append the metadata and dataset DataFrame to the list
                all_data_with_metadata.append((metadata, temp_df))

                # Track the successful link
                successful_links.append(dataset_link)

            except Exception as e:
                print(f"Failed to download dataset from {dataset_link}: {e}")

    # Return the list of tuples containing metadata and DataFrame for each dataset
    return all_data_with_metadata


