# llm/llm_use.py

import pandas as pd
import logging
from typing import Any
from llm.llm_chatbot import LLMChatbot 
from config_loader import config_loader
import os 
import re

logger = logging.getLogger(__name__)


def directly_use_llm_for_answer(data_input, query: str, chatbot: LLMChatbot, chunk_size: int = 500) -> str:
    """
    Use the LLM to analyze multiple datasets and metadata in a file or DataFrame, chunked for token management.

    Args:
        data_input (str or pd.DataFrame): File path of the dataset or a DataFrame containing datasets and metadata.
        query (str): The user query to answer.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.
        chunk_size (int): The number of rows to include in each chunk.

    Returns:
        str: The LLM's answer based on the metadata and dataset content in chunks, without filtering by year.
    """
    try:
        logger.info(f"Received user query: '{query}'")

        # Check if data_input is a file path or DataFrame
        if isinstance(data_input, pd.DataFrame):
            logger.info("Data input is a DataFrame, proceeding with DataFrame processing.")
            data_df = data_input
        elif isinstance(data_input, str):
            # Ensure the file exists and is accessible
            if not os.path.exists(data_input):
                logger.error(f"File not found: {data_input}")
                return f"Error: File not found: {data_input}"

            # Read the file into a DataFrame
            data_df = pd.read_csv(data_input, header=None, skip_blank_lines=False)
        else:
            logger.error("Unsupported data input type. Must be either a DataFrame or a file path.")
            return "Error: Invalid data input type. Must be a DataFrame or a file path."

        # Process each dataset, chunking it by the chunk_size and ensuring the metadata is included
        final_answer = ""
        current_metadata = ""
        current_dataset = []
        in_metadata = False  # Track if we are in the metadata section
        all_metadata = []
        all_datasets = []

        for index, row in data_df.iterrows():
            row_str = ' '.join(str(x) for x in row)

            # Detect metadata (assuming metadata starts with "Dataset Metadata:")
            if row_str.startswith("Dataset Metadata:"):
                # Finalize any existing dataset before starting the new one
                if current_dataset:
                    all_datasets.append((current_metadata, pd.DataFrame(current_dataset)))
                    current_dataset = []  # Reset dataset collection

                # Start processing new metadata
                current_metadata = row_str.replace("Dataset Metadata:", "").strip()
                all_metadata.append(current_metadata)
                in_metadata = True  # Mark we are now in a metadata section

            elif row_str.strip() == "":
                # Empty lines indicate the end of a dataset (or gap between datasets)
                in_metadata = False  # Metadata section ends when we hit an empty line

            else:
                # If not metadata or empty, it's dataset content
                current_dataset.append(row)
        
        # Process the last dataset if it exists
        if current_dataset:
            all_datasets.append((current_metadata, pd.DataFrame(current_dataset)))

        # Now process all datasets with their metadata
        for metadata, dataset in all_datasets:
            final_answer += process_dataset_chunk(metadata, dataset, query, chatbot, chunk_size)

         # Log the final answer
        logger.info(f"Final LLM Answer for query '{query}': {final_answer.strip()}")

        # Return structured final output
        return (
            f"Query: {query}\n\n"
            f"Answer based on the provided datasets:\n{final_answer.strip()}"
        )

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        return "Error: The specified file could not be found."

    except Exception as e:
        logger.error(f"Error in directly_use_llm_for_answer while processing query '{query}': {e}")
        return f"Sorry, I encountered an error while processing your request. Details: {str(e)}"



def process_dataset_chunk(metadata, dataset, query, chatbot, chunk_size):
    """
    Process a dataset in chunks with the associated metadata.

    Args:
        metadata (str): Metadata summary for the dataset.
        dataset (pd.DataFrame): The actual dataset to process.
        query (str): The user query.
        chatbot (LLMChatbot): The chatbot instance to generate responses.
        chunk_size (int): The number of rows in each chunk.

    Returns:
        str: The combined answer for all chunks, structured clearly with metadata and dataset content.
    """
    total_rows = len(dataset)
    logger.info(f"Processing dataset with {total_rows} rows, chunking into {chunk_size}-row parts.")
    
    dataset_answer = ""

    for i in range(0, total_rows, chunk_size):
        # Get the chunk of data
        data_chunk = dataset.iloc[i:i+chunk_size].to_csv(index=False)

        # Structure the LLM prompt for better response generation
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Metadata for this dataset:\n{metadata}\n\n"  # Include metadata
            f"Here is a chunk of the dataset:\n{data_chunk}\n\n"  # Include chunked dataset
            "You are a helpful Chat Bot specialized in answering questions about, discussing, and referencing datasets from open data portals."
            " Please analyze the chunks and answer the query by considering the datasets provi‚ded."
            " Always reference the correct relevant dataset hyperlinks in the beginning once. "
        )

        # Send the prompt to the LLM and get the answer for this chunk
        try:
            chunk_answer = chatbot.generate_response(context=data_chunk, query=prompt)
            dataset_answer += chunk_answer + "\n"
            logger.info(f"Received response for chunk {i // chunk_size + 1}.")
        except Exception as api_error:
            logger.error(f"Error in LLM API call for chunk {i // chunk_size + 1}: {api_error}")
            return f"Error: The LLM encountered an issue while processing chunk {i // chunk_size + 1}. Details: {str(api_error)}"

    return dataset_answer




def use_llm_for_metadata_selection(df: pd.DataFrame, query: str, chatbot: LLMChatbot) -> pd.DataFrame:
    """
    Use the LLM to parse through the metadata summaries and select relevant datasets.

    Args:
        df (pd.DataFrame): Dataframe containing metadata (title, summary, links).
        query (str): The user query to determine relevant datasets.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        pd.DataFrame: A dataframe containing only the relevant datasets based on the LLM's decision.
    """
    relevant_indices = []
    total_datasets = len(df)
    logger.info(f"Starting metadata selection for {total_datasets} datasets.")

    for idx, row in df.iterrows():
        metadata_content = (
            f"Title: {row['title']}\n"
            f"Summary: {row['summary']}\n"
            f"Link: {row['links']}"
        )
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Below is a dataset metadata entry:\n\n{metadata_content}\n\n"
            "Is this dataset relevant to the query? Answer with 'yes' or 'no'."
        )

        try:
            # Ask the LLM if the dataset is relevant
            llm_response = chatbot.generate_response(context=metadata_content, query=prompt)
            logger.debug(f"LLM response for dataset {idx}: {llm_response}")

            if 'yes' in llm_response.lower():
                relevant_indices.append(idx)
                logger.debug(f"Dataset {idx} marked as relevant.")

        except Exception as e:
            logger.error(f"Error processing dataset {idx}: {e}")
            continue  # Skip this dataset and proceed with others

    # Filter the dataframe to include only relevant datasets

    relevant_datasets = df.iloc[relevant_indices].reset_index(drop=True)
    logger.info(f"Metadata selection completed. {len(relevant_datasets)} out of {total_datasets} datasets are relevant.")

    return relevant_datasets

def directly_use_llm_for_follow_up(query: str, refined_datasets: pd.DataFrame, previous_answer: str, chatbot: LLMChatbot, data_df: pd.DataFrame) -> str:
    """
    Process follow-up questions by using the previous answer, relevant datasets, and downloaded datasets as context.
    This function optimizes the follow-up response by also referring to the downloaded datasets if needed.
    """
    # Construct the follow-up prompt, including the previous answer and relevant datasets
    follow_up_prompt = (
        f"Previously, you answered:\n{previous_answer}\n\n"
        f"The user is now asking a follow-up question: '{query}'.\n"
        "Based on the previous answer and the relevant datasets, provide a detailed response."
    )

    # Optionally, include a summary of the datasets in the prompt if necessary
    dataset_summaries = "\n\n".join([f"Dataset Title: {row['title']}\nSummary: {row['summary']}" for idx, row in refined_datasets.iterrows()])

    # Add dataset summaries to the prompt
    prompt = follow_up_prompt + "\n\nHere are the relevant datasets:\n" + dataset_summaries

    # If the user is specifically asking for links or information available in the downloaded datasets, process that.
    if "link" in query.lower() or "url" in query.lower() or "source" in query.lower():
        # If the user asks for links, extract them from the downloaded datasets
        links = data_df.get('links', None)
        if links is not None:
            link_info = "\n".join(links.dropna())  # Combine all the valid links
            prompt += f"\n\nHere are the dataset links that you requested:\n{link_info}"
        else:
            prompt += "\n\nNo links are available in the downloaded datasets."

    # If the user is asking for specific dataset details (e.g., "more details", "summary"), you can handle that similarly
    elif "details" in query.lower() or "summary" in query.lower():
        prompt += "\n\nThe user is asking for more details or summaries from the datasets. Provide further insights based on the dataset content."

    # Send the prompt to the LLM and get the follow-up answer
    try:
        follow_up_answer = chatbot.generate_response(context=previous_answer, query=prompt)
         # Log the follow-up answer
        logger.info(f"Final LLM Follow-Up Answer for query '{query}': {follow_up_answer.strip()}")
        return follow_up_answer.strip()
    except Exception as e:
        logger.error(f"Error processing follow-up question: {e}")
        return f"Error: Could not process follow-up question. {str(e)}"  