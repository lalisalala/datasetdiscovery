import numpy as np

def generate_summary_with_llm(metadata_row, chatbot):
    """
    Generate a summary for a specific dataset using the LLM.
    """
    metadata_content = (
        f"Title: {metadata_row['title']}\n"
        f"Summary: {metadata_row['summary']}\n"
        f"Link: {metadata_row['links']}\n"
        f"Dataset Name: {metadata_row['name']}\n"
    )

    prompt = (
        f"Based on the following dataset metadata, generate a concise summary in 1-2 sentences:\n\n"
        f"{metadata_content}\n\n"
        "Please provide a concise summary including all the mentioned metadata."
    )
    new_summary = chatbot.generate_response(context=metadata_content, query=prompt)
    return new_summary.strip()

def generate_summaries_for_relevant_datasets(relevant_datasets, chatbot):
    """
    Generate summaries only for relevant datasets.
    """
    relevant_datasets['metadatasummary'] = relevant_datasets.apply(
        lambda row: generate_summary_with_llm(row, chatbot), axis=1
    )
    return relevant_datasets


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

