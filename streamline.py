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

    # Perform FAISS search on the basic metadata (no summaries yet)
    k=10
    print(f"Performing FAISS search with top {k} results...")
    combined_text = df['title'] + df['summary'] +  df['name'] + df['links']  # No summaries yet
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

    # Generate metadata summaries only for the relevant datasets
    relevant_datasets_with_summaries = generate_summaries_for_relevant_datasets(relevant_datasets, chatbot)
    print(f"Use LLM to select important Metadata")
    relevant_datasets_with_summaries = use_llm_for_metadata_selection(relevant_datasets_with_summaries, query, chatbot)
    print(f"Relevant datasets selected by the LLM:\n{relevant_datasets_with_summaries[['title', 'links']]}")


    # Download the relevant datasets using the links
    combined_df = download_datasets(relevant_datasets_with_summaries)

    # Load the downloaded datasets (data.csv)
    data_df = pd.read_csv('data.csv')

    # Use the LLM to analyze the data and provide the answer
    print("Analyzing data with LLM...")
    final_answer = directly_use_llm_for_answer(data_df, query, chatbot)

    print(f"Final Answer:\n{final_answer}") 