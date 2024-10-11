# llm/llm_use.py

import pandas as pd
import logging
from typing import Any
from llm.llm_chatbot import LLMChatbot 
from config_loader import config_loader
import os 
import re
from sparql_query import retrieve_audit_data  # Import the SPARQL query function
from rdflib import Graph 

logger = logging.getLogger(__name__)

def directly_use_llm_for_answer(data_input, query: str, chatbot: LLMChatbot, chunk_size: int = 200, additional_context: str = "") -> str:
    """
    Use the LLM to analyze multiple datasets and metadata in a file or DataFrame, chunked for token management.
    Now includes querying the RDF graph to improve accuracy, and dataset links are referenced.
    
    The `additional_context` is used to provide graph-based results (SPARQL results).
    """
    # Assuming `data_input` is a list of tuples [(metadata, DataFrame), ...]
    llm_input = ""
    
    for metadata, df in data_input:
        # Convert metadata dictionary to a formatted string
        metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])

        # Convert the DataFrame to a CSV string
        data_str = df.to_csv(index=False)

        # Append both metadata and dataset to the input
        llm_input += f"Metadata:\n{metadata_str}\n\nData:\n{data_str}\n\n"

    # Prepare the LLM prompt
    if additional_context:
        llm_prompt = (
            f"User query: {query}\n\n"
            f"Based on the knowledge graph, here are the relevant datasets and audits:\n{additional_context}\n\n"
            "Please analyze the dataset contents and provide a detailed response."
            f"\n\nMetadata for reference (including dataset links):\n{llm_input}"  # Include metadata with links
        )
    else:
        llm_prompt = (
            f"User query: {query}\n\n"
            "Please proceed using the datasets directly."
            f"\n\nMetadata for reference (including dataset links):\n{llm_input}"  # Include metadata with links
        )

    # Use the LLM to generate a final response
    try:
        final_llm_answer = chatbot.generate_response(context=llm_input, query=llm_prompt)
        
        # Log the final LLM answer
        logger.info(f"Final LLM Answer for query '{query}':\n{final_llm_answer.strip()}")

        return final_llm_answer.strip()

    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return f"Error generating response: {str(e)}"



def process_dataset_chunk(metadata, dataset, query, chatbot, chunk_size):
    """
    Process a dataset in chunks with the associated metadata.

    Args:
        metadata (str): Metadata summary for the dataset.
        dataset (pd.DataFrame): The actual dataset to process.
        query (str): The user query.
        chatbot (LLMChatbot): The chatbot instance to generate responses.
        chunk_size (int): The number of rows to include in each chunk.

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
            " Please analyze the chunks and answer the query by considering the datasets provided."
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

from sparql_query import retrieve_audit_data  # Import the SPARQL query function

def directly_use_llm_for_follow_up(query: str, refined_datasets: pd.DataFrame, previous_answer: str, chatbot: LLMChatbot, data_df: pd.DataFrame) -> str:
    """
    Process follow-up questions by using the previous answer, relevant datasets, and downloaded datasets as context.
    This function optimizes the follow-up response by referring to the RDF graph first, and then using the LLM.
    """
    # Step 1: Query the RDF graph for relevant information (related to the follow-up question)
    sparql_results = retrieve_audit_data(query)  # Query the RDF knowledge graph based on the follow-up query

    # Step 2: Process the SPARQL results (if any) and integrate them into the follow-up response
    graph_answer = ""
    if sparql_results:
        for row in sparql_results:
            dataset, audit, scope, link = row  # Fetch link from the RDF query result
            graph_answer += f"Dataset: {dataset}\nAudit: {audit}\nScope: {scope}\nLink: {link}\n\n"  # Include the link

        # Use the graph-based information as context for the follow-up question
        follow_up_prompt = (
            f"Previously, you answered:\n{previous_answer}\n\n"
            f"The user is now asking a follow-up question: '{query}'.\n"
            f"Based on the knowledge graph, here are the relevant datasets and audits:\n{graph_answer}\n\n"
            "Please provide a detailed response considering the datasets and the user's follow-up question."
        )
    else:
        # If no relevant data was found in the RDF graph, use the previous answer and dataset context
        follow_up_prompt = (
            f"Previously, you answered:\n{previous_answer}\n\n"
            f"The user is now asking a follow-up question: '{query}'.\n"
            "Based on the previous answer and the relevant datasets, provide a detailed response."
        )

    # Optionally: Include a summary of the datasets in the prompt if necessary
    dataset_summaries = "\n\n".join([f"Dataset Title: {row['title']}\nSummary: {row['summary']}\nLink: {row['links']}" for idx, row in refined_datasets.iterrows()])
    follow_up_prompt += "\n\nHere are the relevant datasets:\n" + dataset_summaries

    # If the user is asking for links or information from the downloaded datasets, include that in the prompt
    if "link" in query.lower() or "url" in query.lower() or "source" in query.lower():
        links = data_df.get('links', None)
        if links is not None:
            link_info = "\n".join(links.dropna())  # Combine all the valid links
            follow_up_prompt += f"\n\nHere are the dataset links that you requested:\n{link_info}"
        else:
            follow_up_prompt += "\n\nNo links are available in the downloaded datasets."

    # Step 3: Use the LLM to generate a final response
    try:
        follow_up_answer = chatbot.generate_response(context=previous_answer, query=follow_up_prompt)
        
        # Log the final follow-up LLM answer
        logger.info(f"Final LLM Follow-Up Answer for query '{query}':\n{follow_up_answer.strip()}")

        return follow_up_answer.strip()
    except Exception as e:
        logger.error(f"Error processing follow-up question: {e}")
        return f"Error: Could not process follow-up question. {str(e)}"
