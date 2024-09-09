import os
import subprocess
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from faiss_index import create_faiss_index, search_faiss, load_data
from llm_chatbot import LLMChatbot

def run_webscraping():
    print("Running web scraping...")
    subprocess.run(['python', 'webscraping.py'], check=True)

def create_faiss_index_from_csv(csv_file='datasets.csv'):
    df, metadata, content = load_data(csv_file)
    model, metadata_index, content_index, content_embeddings = create_faiss_index(metadata, content)
    return model, metadata_index, content_index, content_embeddings, df

def query_faiss_index(query, model, metadata_index, content_index, content_embeddings, df, k=5):
    # Search in metadata index
    indices_metadata, distances_metadata, _, _ = search_faiss(
        query, model, metadata_index, content_index, content_embeddings, k
    )
    
    # Extract relevant dataset content using metadata search results
    relevant_datasets_indices = indices_metadata
    relevant_contents = df.iloc[relevant_datasets_indices]['content'].tolist()
    
    # Create a new FAISS index for the relevant contents
    content_model = SentenceTransformer('all-MiniLM-L6-v2')
    content_embeddings_relevant = content_model.encode(relevant_contents)
    content_index_relevant = faiss.IndexFlatL2(content_embeddings_relevant.shape[1])
    content_index_relevant.add(np.array(content_embeddings_relevant))
    
    # Prepare the query embedding
    query_embedding = model.encode([query])
    
    # Ensure query_embedding is in the correct shape: (1, d)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Perform FAISS search on the relevant content
    indices_content_relevant, distances_content_relevant = content_index_relevant.search(query_embedding, k)
    
    return indices_metadata, distances_metadata, indices_content_relevant, distances_content_relevant

def main():
    csv_file = 'datasets.csv'
    webscraping_file = 'webscraping.py'
    
    # Run web scraping if the CSV file is outdated
    if not os.path.isfile(csv_file) or os.path.getmtime(csv_file) < os.path.getmtime(webscraping_file):
        run_webscraping()
    
    # Create FAISS index
    model, metadata_index, content_index, content_embeddings, df = create_faiss_index_from_csv(csv_file)
    
    # Define the query
    query = "How many audits were planned in 2016-2017? Can you name them? "
    
    # Perform FAISS search
    indices_metadata, distances_metadata, indices_content_relevant, distances_content_relevant = query_faiss_index(
        query, model, metadata_index, content_index, content_embeddings, df
    )
    
    # Initialize contexts
    context_metadata = "No metadata results found."
    context_content = "No content results found."
    
    # Construct metadata context if results are found
    if len(indices_metadata) > 0:
        context_metadata = (
            f"Metadata Results:\n"
            f"Indices: {indices_metadata}\n"
            f"Distances: {distances_metadata}\n\n"
            f"Based on the above information, please answer the query: '{query}'"
        )
        print("Metadata results:")
        print(f"Indices: {indices_metadata}\nDistances: {distances_metadata}")
    
    # Construct content context if results are found
    if len(indices_content_relevant) > 0:
        context_content = (
            f"Content Results:\n"
            f"Indices: {indices_content_relevant}\n"
            f"Distances: {distances_content_relevant}\n\n"
            f"Based on the above content, please answer the query: '{query}'"
        )
        print("Content results:")
        print(f"Indices: {indices_content_relevant}\nDistances: {distances_content_relevant}")
    
    # Initialize chatbot
    chatbot = LLMChatbot(model_name='mistral')
    
    # Generate responses
    response_metadata = chatbot.generate_response(context_metadata, query)
    response_content = chatbot.generate_response(context_content, query)
    
    # Print responses
    print("Generated Response for Metadata:")
    print(response_metadata)
    print("Generated Response for Content:")
    print(response_content)

if __name__ == "__main__":
    main()