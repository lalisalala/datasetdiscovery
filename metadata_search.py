import faiss
import numpy as np

def generate_summaries_for_datasets(df, llm_chatbot):
    """
    Generate new summaries for all datasets using the LLM and return a dataframe
    with the metadata summaries and links.

    Args:
        df (pd.DataFrame): The dataframe containing dataset metadata (title, summary, links).
        llm_chatbot (LLMChatbot): The LLMChatbot instance used to generate summaries.

    Returns:
        pd.DataFrame: The dataframe with the new 'metadatasummary' and 'links' columns.
    """
    df['metadatasummary'] = df.apply(lambda row: generate_summary_with_llm(row, llm_chatbot), axis=1)
    
    # Keep only the metadata summary and links in the final dataset
    return df[['title', 'metadatasummary', 'links']]

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

