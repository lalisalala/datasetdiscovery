import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def create_faiss_index_for_data(data):
    """
    Create a FAISS index for the content in the downloaded datasets stored in data.csv.
    
    Args:
        data (list): A list of text data (rows or specific columns) from data.csv to index.
        
    Returns:
        model: The SentenceTransformer model used for encoding.
        index: The FAISS index for the data.
    """
    # Use a pre-trained transformer model to encode text data
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data)
    
    # Create a FAISS index with the embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return model, index

def search_data(query, model, index, data, k=5):
    """
    Perform a FAISS search on the data content.
    
    Args:
        query (str): The user query to search for.
        model: The SentenceTransformer model used for encoding the query.
        index: The FAISS index for the data.
        data (list): The original data to retrieve results from.
        k (int): The number of top results to return.
        
    Returns:
        list: A list of top-k data results that match the query.
    """
    # Encode the query and perform a FAISS search
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the top-k relevant results
    results = [data[idx] for idx in indices[0]]
    return results

def identify_text_columns(df):
    """
    Identify columns in the dataframe that contain text (string-based columns).
    
    Args:
        df (pd.DataFrame): The dataframe containing the downloaded data (data.csv).
        
    Returns:
        list: A list of column names that are text-based.
    """
    text_columns = []
    for col in df.columns:
        # Check if the column contains object data type (usually string/text data)
        if df[col].dtype == 'object':
            text_columns.append(col)
    return text_columns