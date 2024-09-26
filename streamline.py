
import os
import pandas as pd
from search.metadata_search import query_faiss_index, generate_summaries_for_relevant_datasets
from search.faiss_index import create_faiss_index
from search.data_search import download_datasets
from llm.llm_chatbot import LLMChatbot
from llm.llm_use import directly_use_llm_for_answer, use_llm_for_metadata_selection
from web.webscraping import run_webscraping
import time
import logging
from config_loader import config_loader

logger = logging.getLogger(__name__)

def run_streamline_process(query: str) -> str:
    """
    Modified streamline process to integrate with FastAPI.
    It accepts a query string as an argument.
    """
    logger.info(f"Received user query: '{query}'")

    start_time = time.time()  # Track start time

    # Define the CSV file for saving the datasets
    csv_file = 'datasets.csv'

    # Always run web scraping to overwrite datasets.csv
    logger.info("Running web scraping to update datasets.csv.")
    run_webscraping()

    # Load the dataset metadata from the CSV file (without metadata summaries)
    if not os.path.exists(csv_file):
        logger.error(f"CSV file {csv_file} not found after web scraping.")
        return "Error: Datasets not found."
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} datasets from {csv_file}.")

    # Initialize the LLM chatbot with configurations
    llm_config = config_loader.get_llm_config()
    faiss_config = config_loader.get_faiss_config()

    chatbot = LLMChatbot(
        model_name=llm_config.get('model_name', 'mistral'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 1024),
        api_url=llm_config.get('api_url', 'http://localhost:11434/api/generate')
    )

    # Use the LLM to dynamically determine the number of top results (k) for FAISS
    k = faiss_config.get('top_k', 10)

    # Perform FAISS search on the basic metadata (no summaries yet)
    logger.info(f"Performing FAISS search with top {k} results...")
    combined_text = df['title'] + " " + df['links']  # No summaries yet
    model, metadata_index = create_faiss_index(combined_text.tolist())
    best_indices, best_distances = query_faiss_index(query, model, metadata_index, k)

    # Ensure indices are within bounds of the DataFrame
    valid_indices = [i for i in best_indices if i < len(df)]
    if not valid_indices:
        logger.warning("No valid indices found from FAISS search.")
        return "No relevant datasets found."

    # Retrieve relevant datasets based on valid FAISS indices
    relevant_datasets = df.iloc[valid_indices]
    logger.info(f"Best Metadata Results (Distances: {best_distances}):\n{relevant_datasets[['title', 'links']]}")

    # Save the FAISS search results to datasets2.csv
    datasets2_csv = 'datasets2.csv'
    relevant_datasets.to_csv(datasets2_csv, index=False)
    logger.info(f"Saved FAISS search results to {datasets2_csv}.")

    # Load the results from datasets2.csv
    df_faiss_results = pd.read_csv(datasets2_csv)

    # Use the LLM to further refine the FAISS results and find the most relevant datasets
    refined_datasets = use_llm_for_metadata_selection(df_faiss_results, query, chatbot)
    logger.info(f"Refined to {len(refined_datasets)} relevant datasets after LLM processing.")

    # Generate metadata summaries for the found datasets
    refined_datasets_with_summaries = generate_summaries_for_relevant_datasets(refined_datasets, chatbot)
    logger.info("Generated metadata summaries for refined datasets.")

    # Download the final refined datasets using the links
    successful_links = []
    combined_df = download_datasets(refined_datasets_with_summaries, output_file='data.csv')

    # Load the downloaded datasets (data.csv)
    data_csv = 'data.csv'
    if not os.path.exists(data_csv):
        logger.error(f"Data file {data_csv} not found after downloading datasets.")
        return "Error: Downloaded data not found."

    data_df = pd.read_csv(data_csv)
    logger.info(f"Loaded downloaded data from {data_csv} with {len(data_df)} records.")

    # Use the LLM to analyze the data and provide the answer
    logger.info("Directly analyzing with LLM...")
    final_answer = directly_use_llm_for_answer(data_df, query, chatbot)
    logger.info(f"Final Answer:\n{final_answer}")

    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time

    # Log execution details to a file
    logger.info(f"Streamline process completed in {total_time:.2f} seconds.")

    return final_answer  # Return the final answer for the API
