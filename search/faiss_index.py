import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_data(csv_file='datasets.csv'):
    """
    Load the dataset from CSV and use the 'generated_summary' column for FAISS indexing.
    Ensure that the LLM-generated summaries are already present in the 'datasets.csv'.
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Use the 'generated_summary' column for FAISS indexing
    if 'generated_summary' not in df.columns:
        raise ValueError("The 'generated_summary' column is missing. Run the LLM summary generation first.")
    
    # Use generated summaries for FAISS indexing
    metadata = df['generated_summary'].tolist()
    
    return df, metadata

def create_faiss_index(metadata):
    """
    Create a FAISS index using the embeddings of the LLM-generated summaries.
    """
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the metadata (summaries) into embeddings
    metadata_embeddings = model.encode(metadata)
    
    # Create a FAISS index for the embeddings
    dimension = metadata_embeddings.shape[1]
    metadata_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the FAISS index
    metadata_index.add(np.array(metadata_embeddings))
    
    return model, metadata_index

def search_faiss(query, model, metadata_index, k=20):
    """
    Perform a search query using the FAISS index based on the LLM-generated summaries.
    """
    # Encode the query into an embedding
    query_embedding = model.encode([query])
    
    # Perform the FAISS search
    metadata_distances, metadata_indices = metadata_index.search(np.array(query_embedding), k)
    
    return metadata_indices[0], metadata_distances[0]

if __name__ == "__main__":
    # Load the data and LLM-generated summaries
    df, metadata = load_data('datasets.csv')
    
    # Create the FAISS index using the generated summaries
    model, metadata_index = create_faiss_index(metadata)
    
    # Perform a search query
    query = "Your search query here"
    indices, distances = search_faiss(query, model, metadata_index)
    
    # Display the search results
    results = df.iloc[indices]
    print(results[['title', 'generated_summary', 'links']])
