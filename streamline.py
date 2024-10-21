import os
import pandas as pd
from search.metadata_search import query_faiss_index, generate_summaries_for_relevant_datasets
from search.faiss_index import create_faiss_index
from search.data_search import download_datasets
from llm.llm_chatbot import LLMChatbot
from llm.llm_use import directly_use_llm_for_answer, use_llm_for_metadata_selection, directly_use_llm_for_follow_up
from web.webscraping import run_webscraping
import time
import logging
from config_loader import config_loader
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Global dictionary for storing user sessions and context
user_sessions = {}

def run_streamline_process(query: str, user_id: str) -> str:
    """
    Automatically detect whether the query is a follow-up question or a new query.
    """

    # Retrieve the user's session context
    user_context = user_sessions.get(user_id, {})

    # Automatically detect if it's a follow-up question
    follow_up = False  # Default to False (assume it's a new query)

    # Check if there is previous context for this user
    if user_context:
        # If we have context, check for relevant keywords or signs of a follow-up
        previous_answer = user_context.get('previous_answer')
        if previous_answer:
            # Simple keyword-based check to detect follow-up queries
            follow_up_keywords = ["more information", "details", "clarify", "explain", "about dataset", "continue", "expand", "again", "specify", "link"]
            
            # Check if the query mentions follow-up keywords or if it's similar to the previous answer
            if any(keyword in query.lower() for keyword in follow_up_keywords):
                follow_up = True
            elif check_similarity(query, previous_answer):  # Optional: you can implement a query similarity check
                follow_up = True

        # If follow_up is true, use the existing context to handle the query
        if follow_up:
            refined_datasets = user_context.get('relevant_datasets')
            if refined_datasets is not None:
                # Load the previously downloaded datasets
                if os.path.exists('data.csv'):
                    data_df = pd.read_csv('data.csv')
                else:
                    return "Error: Downloaded data not found for follow-up question."
                
                # Call the follow-up processing function and pass the datasets
                return directly_use_llm_for_follow_up(query, refined_datasets, previous_answer, user_context['chatbot'], data_df)
            else:
                return "Error: No relevant datasets found in the context for follow-up."
    
    # If not a follow-up, treat it as a new query and process the entire pipeline
    return process_new_query(query, user_id)



def process_new_query(query: str, user_id: str) -> str:
    """
    Handle a new query by running the full pipeline and saving the context.
    """
    logger.info(f"Received new user query: '{query}'")

    start_time = time.time()  # Track start time
    csv_file = 'datasets.csv'

    # Always run web scraping to update datasets.csv
    logger.info("Running web scraping to update datasets.csv.")
    run_webscraping()

    if not os.path.exists(csv_file):
        logger.error(f"CSV file {csv_file} not found after web scraping.")
        return "Error: Datasets not found."
    
    # Attempt to read the CSV with error handling using on_bad_lines parameter
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python', sep=',')
    except pd.errors.ParserError as e:
        logger.error(f"Error reading the CSV file: {e}")
        return f"Error: Failed to read the datasets.csv file due to parsing issues. Details: {e}"
    
    llm_config = config_loader.get_llm_config()
    faiss_config = config_loader.get_faiss_config()

    chatbot = LLMChatbot(
        model_name=llm_config.get('model_name', 'mistral'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 1024),
        api_url=llm_config.get('api_url', 'http://localhost:11434/api/generate')
    )

    # Use FAISS to find relevant datasets based on the query
    k = faiss_config.get('top_k', 10)
    combined_text = df['title'] + " " + df['links']
    model, metadata_index = create_faiss_index(combined_text.tolist())
    best_indices, best_distances = query_faiss_index(query, model, metadata_index, k)

    valid_indices = [i for i in best_indices if i < len(df)]
    if not valid_indices:
        return "No relevant datasets found."

    # Retrieve relevant datasets and process further
    relevant_datasets = df.iloc[valid_indices]
    datasets2_csv = 'datasets2.csv'
    relevant_datasets.to_csv(datasets2_csv, index=False)

    try:
        df_faiss_results = pd.read_csv(datasets2_csv)
    except pd.errors.ParserError as e:
        logger.error(f"Error reading FAISS results CSV: {e}")
        return f"Error: Failed to read datasets2.csv file due to parsing issues."

    refined_datasets = use_llm_for_metadata_selection(df_faiss_results, query, chatbot)
    refined_datasets_with_summaries = generate_summaries_for_relevant_datasets(refined_datasets, chatbot)
    download_datasets(refined_datasets_with_summaries, output_file='data.csv')

    if not os.path.exists('data.csv'):
        return "Error: Downloaded data not found."

    # Attempt to read the final dataset with error handling
    try:
        data_df = pd.read_csv('data.csv', on_bad_lines='skip', engine='python', sep=',')
    except pd.errors.ParserError as e:
        logger.error(f"Error reading final data.csv: {e}")
        return f"Error: Failed to read the data.csv file due to parsing issues. Details: {e}"

    final_answer = directly_use_llm_for_answer(data_df, query, chatbot)

    # Save the context for follow-up questions
    user_sessions[user_id] = {
        'relevant_datasets': refined_datasets_with_summaries,
        'previous_answer': final_answer,
        'chatbot': chatbot  # Store chatbot instance for follow-up
    }

    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Streamline process completed in {total_time:.2f} seconds.")

    return final_answer

def check_similarity(query: str, previous_answer: str) -> bool:
    """
    Optional: Implement a simple string similarity check to see if the query is a follow-up.
    You can enhance this with more sophisticated natural language processing.
    """
    # Use basic string similarity to check if the query is a follow-up
    similarity_ratio = SequenceMatcher(None, query, previous_answer).ratio()
    
    # Consider it a follow-up if the similarity ratio is above a threshold (e.g., 0.3)
    return similarity_ratio > 0.3
