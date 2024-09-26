# llm/llm_use.py

import pandas as pd
import logging
from typing import Any
from llm.llm_chatbot import LLMChatbot 
from config_loader import config_loader
import os 

logger = logging.getLogger(__name__)

def directly_use_llm_for_answer(data_input, query: str, chatbot: LLMChatbot) -> str:
    """
    Use the LLM to directly analyze data from a DataFrame or a file.
    The LLM is prompted to reference the metadata in its response.

    Args:
        data_input (str or pd.DataFrame): Either the file path of the dataset or the dataset as a DataFrame.
        query (str): The user query to answer.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        str: The LLM's answer based on the metadata and dataset content in the file and the query.
    """
    try:
        logger.info(f"Received user query: '{query}'")

        # Check if data_input is a file path or DataFrame
        if isinstance(data_input, pd.DataFrame):
            logger.info("Data input is a DataFrame, proceeding with DataFrame processing.")
            data_content = data_input.to_csv(index=False)
        elif isinstance(data_input, str):
            # Ensure the file exists and is accessible
            if not os.path.exists(data_input):
                logger.error(f"File not found: {data_input}")
                return f"Error: File not found: {data_input}"

            # Open the file and read the contents as text
            with open(data_input, 'r') as f:
                data_content = f.read()
        else:
            logger.error("Unsupported data input type. Must be either a DataFrame or a file path.")
            return "Error: Invalid data input type. Must be a DataFrame or a file path."

        # Log the file or DataFrame contents (first few lines for debugging)
        logger.debug(f"Data content (first 500 chars): {data_content[:500]}")

        # Construct the LLM prompt
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Here are the relevant datasets along with their metadata summaries:\n{data_content}\n\n"
            "Please use the metadata and dataset content to generate a detailed and accurate answer to the user's query. "
            "Ensure that the metadata (including the link) are referenced for context."
        )
        logger.info("Prompt constructed for directly_use_llm_for_answer.")
        logger.debug(f"Prompt (first 500 chars): {prompt[:500]}")

        # Send the prompt to the LLM and get the answer
        try:
            final_answer = chatbot.generate_response(context=data_content, query=prompt)
            logger.info("Received response from LLM.")
            logger.debug(f"LLM response (first 500 chars): {final_answer[:500]}")

        except Exception as api_error:
            logger.error(f"Error in LLM API call: {api_error}")
            return f"Error: The LLM encountered an issue while processing your request. Details: {str(api_error)}"

        return final_answer

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        return "Error: The specified file could not be found."

    except Exception as e:
        logger.error(f"Error in directly_use_llm_for_answer while processing query '{query}': {e}")
        return f"Sorry, I encountered an error while processing your request. Details: {str(e)}"




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

