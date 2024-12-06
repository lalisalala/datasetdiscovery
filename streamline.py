import os
import pandas as pd
from search.metadata_search import query_faiss_index, generate_summaries_for_relevant_datasets
from search.faiss_index import create_faiss_index
from search.data_search import download_datasets
from llm.llm_chatbot import LLMChatbot
from llm.llm_use import directly_use_llm_for_answer, use_llm_for_metadata_selection, directly_use_llm_for_follow_up
from datastore_api_access import run_datastore_access  # Updated import
from sparql_query import retrieve_metadata
from graph import generate_dynamic_rdf_with_core
import time
import logging
from config_loader import config_loader
from difflib import SequenceMatcher
import re

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary for storing user sessions and context
user_sessions = {}


def postprocess_metadata(metadata_list):
    """
    Post-process the metadata list to clean formatting issues and ensure consistency.
    Args:
        metadata_list (list): List of metadata dictionaries for datasets.

    Returns:
        list: Cleaned and standardized metadata list.
    """
    def clean_text(value):
        """Clean text by removing unwanted characters and formatting artifacts."""
        if not isinstance(value, str):
            return value
        # Remove newlines, tabs, and carriage returns
        cleaned_value = re.sub(r'[\r\n\t]+', ' ', value)
        # Collapse multiple spaces into a single space
        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
        # Remove known artifacts
        artifacts = [
            "Normal 0", "false false false", "MicrosoftInternetExplorer4", 
            "/\\* Style Definitions \\*/", "table.MsoNormalTable"
        ]
        for artifact in artifacts:
            cleaned_value = cleaned_value.replace(artifact, '')
        return cleaned_value.strip()

    # Process each dataset's metadata
    cleaned_metadata_list = []
    for metadata in metadata_list:
        cleaned_metadata = {key: clean_text(value) for key, value in metadata.items()}
        # Additional processing for specific fields (if needed)
        if 'topic' in cleaned_metadata and cleaned_metadata['topic']:
            # Ensure topics are consistently comma-separated
            cleaned_metadata['topic'] = ", ".join([t.strip() for t in cleaned_metadata['topic'].split(',')])
        cleaned_metadata_list.append(cleaned_metadata)

    return cleaned_metadata_list


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
            follow_up_keywords = ["more information", "details", "clarify", "explain", "about dataset", "continue", "expand", "again", "specify", "link"]

            # Check if the query mentions follow-up keywords or if it's similar to the previous answer
            if any(keyword in query.lower() for keyword in follow_up_keywords):
                follow_up = True
            elif check_similarity(query, previous_answer):
                follow_up = True

        # If follow_up is true, use the existing context to handle the query
        if follow_up:
            refined_datasets = user_context.get('relevant_datasets')
            all_data = user_context.get('all_data_with_metadata')
            if refined_datasets is not None and all_data is not None:
                # Call the follow-up processing function and pass the datasets
                return directly_use_llm_for_follow_up(query, refined_datasets, previous_answer, user_context['chatbot'], all_data)
            else:
                return "Error: No previous datasets found in the context for follow-up."

    # If not a follow-up, treat it as a new query and process the entire pipeline
    return process_new_query(query, user_id)


def process_new_query(query: str, user_id: str) -> str:
    """
    Handle a new query by running the full pipeline and saving the context.
    The pipeline now generates RDF based solely on metadata and does not download dataset content.
    """
    logger.info(f"Received new user query: '{query}'")

    start_time = time.time()  # Track start time
    csv_file = 'datasets.csv'

    # Always run the API data fetch to update datasets.csv
    logger.info("Running datastore API access to fetch datasets and update datasets.csv.")
    run_datastore_access()  # Updated call to fetch data via API

    if not os.path.exists(csv_file):
        logger.error(f"CSV file {csv_file} not found after datastore API access.")
        return "Error: Datasets not found."

    df = pd.read_csv(csv_file)

    logger.info(f"Acquired metadata for {len(df)} datasets. Listing datasets:")
    for idx, row in df.iterrows():
        logger.info(f"Dataset {idx + 1}: {row['title']} | Publisher: {row['publisher']} | Topic: {row['topic']}")

    llm_config = config_loader.get_llm_config()
    faiss_config = config_loader.get_faiss_config()

    chatbot = LLMChatbot(
        model_name=llm_config.get('model_name', 'mistral'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 1024),
        api_url=llm_config.get('api_url', 'http://localhost:11434/api/generate')
    )

    # Step 1: Use FAISS to find relevant datasets based on the query
    k = faiss_config.get('top_k', 10)
    combined_text = df['title'] + " " + df['links']
    model, metadata_index = create_faiss_index(combined_text.tolist())
    best_indices, best_distances = query_faiss_index(query, model, metadata_index, k)

    valid_indices = [i for i in best_indices if i < len(df)]
    if not valid_indices:
        logger.info("No relevant datasets found for the given query.")
        return "No relevant datasets found."

    # Step 2: Retrieve relevant datasets and process them further
    relevant_datasets = df.iloc[valid_indices]

    # Save the relevant datasets to a CSV file for inspection
    relevant_datasets.to_csv("datasets2.csv", index=False)
    logger.info(f"Saved {len(relevant_datasets)} relevant datasets to 'datasets2.csv'.")

    # Convert to a list of metadata dictionaries
    metadata_list = relevant_datasets.to_dict(orient='records')  # Convert to list of metadata dictionaries
    
    # Post-process the metadata to clean formatting issues
    metadata_list = postprocess_metadata(metadata_list)

    # Step 3: Generate the RDF knowledge graph based on metadata
    generate_dynamic_rdf_with_core(metadata_list, output_rdf_file='metadata_ontology.ttl')

    # Step 4: Query the RDF knowledge graph using SPARQL based on the user's query
    try:
        sparql_results = retrieve_metadata(query)
    except FileNotFoundError:
        logger.error("RDF file 'metadata_ontology.ttl' not found. Ensure the RDF graph is generated.")
        return "Error: RDF knowledge graph not found. Ensure the RDF graph is generated before querying."

    # Step 5: Use SPARQL query results to provide input to the LLM
    if sparql_results:
        graph_answer = ""
        for row in sparql_results:
            dataset, audit, scope, link = row
            graph_answer += f"Dataset: {dataset}\nAudit: {audit}\nScope: {scope}\nLink: {link}\n\n"

        # Use the graph-based context
        final_answer = directly_use_llm_for_answer(metadata_list, query, chatbot, additional_context=graph_answer)
    else:
        # No SPARQL results; proceed without additional context
        final_answer = directly_use_llm_for_answer(metadata_list, query, chatbot)

    # Step 6: Save the context for follow-up questions
    user_sessions[user_id] = {
        'relevant_datasets': metadata_list,
        'previous_answer': final_answer,
        'chatbot': chatbot,  # Store chatbot instance for follow-up
    }

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
