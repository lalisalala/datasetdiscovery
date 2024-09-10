import faiss
import numpy as np

def query_faiss_index(query, model, metadata_index, k=5):
    """
    Perform a FAISS search to find the top-k most relevant metadata.
    
    Args:
        query (str): The search query.
        model: The SentenceTransformer model used for encoding.
        metadata_index: The FAISS index to search.
        k (int): The number of top results to return.
    
    Returns:
        indices_metadata: Indices of the top k results.
        distances_metadata: Distances of the top k results from the query.
    """
    try:
        # Encode the query into an embedding
        query_embedding = model.encode([query])

        # Perform the FAISS search
        distances, indices = metadata_index.search(np.array(query_embedding), k)

        # Return the top k results
        return indices[0], distances[0]
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        raise