import os
import pandas as pd
from search.metadata_search import query_faiss_index, generate_summaries_for_relevant_datasets
from search.faiss_index import create_faiss_index
from search.data_search import download_datasets, load_query_from_yaml
from llm.llm_chatbot import LLMChatbot
from llm.llm_use import directly_use_llm_for_answer, use_llm_for_metadata_selection
from search.data_analysis import create_faiss_index_for_data, search_data
import subprocess
from decision_logic import decide_faiss_or_llm, decide_faiss_or_llm_for_metadata, identify_text_columns
from web.webscraping import run_webscraping
import yaml 

def main():
    # Load the user query from the YAML file
    query = load_query_from_yaml()

    # Define the CSV file for saving the datasets
    csv_file = 'datasets.csv'

    # Always run web scraping to overwrite datasets.csv
    run_webscraping()

    # Load the dataset metadata from the CSV file (without metadata summaries)
    df = pd.read_csv(csv_file)

    # Initialize the LLM chatbot for relevance checking
    chatbot = LLMChatbot(model_name='mistral')

    # Use the LLM to dynamically determine the number of top results (k) for FAISS
    k = 10

    # Perform FAISS search on the basic metadata (no summaries yet)
    print(f"Performing FAISS search with top {k} results...")
    combined_text = df['title'] + " " + df['links']  # No summaries yet
    model, metadata_index = create_faiss_index(combined_text.tolist())
    best_indices, best_distances = query_faiss_index(query, model, metadata_index, k)

    # Ensure indices are within bounds of the DataFrame
    valid_indices = [i for i in best_indices if i < len(df)]
    if not valid_indices:
        print("No valid indices found. Exiting.")
        return

    # Retrieve relevant datasets based on valid FAISS indices
    relevant_datasets = df.iloc[valid_indices]
    print(f"Best Metadata Results (Distances: {best_distances}):\n")
    print(relevant_datasets[['title', 'links']])

    # Save the FAISS search results to datasets2.csv
    relevant_datasets.to_csv('datasets2.csv', index=False)
    print("Saved FAISS search results to datasets2.csv.")

    # Load the results from datasets2.csv
    df_faiss_results = pd.read_csv('datasets2.csv')

    
    # Use the LLM to further refine the FAISS results and find the most relevant datasets
    refined_datasets = use_llm_for_metadata_selection(df_faiss_results, query, chatbot)

    # Generate metadatasummaries for the found datasets
    refined_datasets_with_summaries = generate_summaries_for_relevant_datasets(refined_datasets, chatbot)


    # Download the final refined datasets using the links
    combined_df = download_datasets(refined_datasets_with_summaries, output_file='data.csv')


    # Load the downloaded datasets (data.csv)
    data_df = pd.read_csv('data.csv')

    
    # If FAISS is not needed, directly use the LLM to analyze the data and provide the answer
    print("Directly analyzing with LLM...")
    final_answer = directly_use_llm_for_answer(data_df, query, chatbot)

    print(f"Final Answer:\n{final_answer}") 