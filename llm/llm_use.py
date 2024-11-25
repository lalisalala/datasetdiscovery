import pandas as pd
import logging
from typing import Any
from llm.llm_chatbot import LLMChatbot
from config_loader import config_loader
from sparql_query import retrieve_audit_data, query_rdf_graph, classify_query, build_dynamic_sparql_query

logger = logging.getLogger(__name__)

def directly_use_llm_for_answer(data_input, query: str, chatbot: LLMChatbot, chunk_size: int = 200, additional_context: str = "") -> str:
    """
    Use the LLM to analyze multiple datasets and metadata in a file or DataFrame, chunked for token management.
    Now includes querying the RDF graph to improve accuracy, and dataset links are referenced.
    
    For broader queries about available datasets, limits response to FAISS search results.
    """
    # Check if query is asking about available datasets
    broad_query_keywords = ["what datasets", "available datasets", "do you have datasets on", "datasets on"]
    if any(keyword in query.lower() for keyword in broad_query_keywords):
        # Directly respond with dataset titles, summaries, and links from FAISS results without deeper processing
        dataset_overview = []
        for metadata, _ in data_input:
            dataset_info = f"Title: {metadata.get('title', 'N/A')}\nSummary: {metadata.get('summary', 'No summary provided')}"
            if 'links' in metadata:
                dataset_info += f"\nLink: {metadata['links']}"
            dataset_overview.append(dataset_info)
        
        # Combine overview of datasets into a single response
        return "\n\n".join(dataset_overview)
    
    # If it’s a specific query, continue with the usual process of dataset analysis
    llm_input = ""
    all_links = []  # Store all dataset links for inclusion later if necessary
    
    for metadata, df in data_input:
        # Convert metadata dictionary to a formatted string
        metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])

        # Store links for later use
        if 'links' in metadata:
            metadata_str += f"\nLink: {metadata['links']}"
            all_links.append(metadata['links'])  # Collect all the links for post-processing

        # Convert the DataFrame to a CSV string
        data_str = df.to_csv(index=False)

        # Append both metadata and dataset to the input
        llm_input += f"Metadata:\n{metadata_str}\n\nData:\n{data_str}\n\n"

    # Prepare the LLM prompt with additional context if provided
    llm_prompt = (
        f"User query: {query}\n\n"
        f"{'Based on the knowledge graph, here are the relevant datasets and audits:\n' + additional_context if additional_context else ''}"
        "\n\nPlease analyze the dataset contents and provide a detailed response, including dataset links."
        f"\n\nMetadata for reference (including dataset links):\n{llm_input}"  # Include metadata with links
    )

    # Log the final prompt for debugging
    logger.debug(f"Final LLM Prompt:\n{llm_prompt}")

    # Use the LLM to generate a final response
    try:
        final_llm_answer = chatbot.generate_response(context=llm_input, query=llm_prompt)
        
        # Post-process the response to ensure all dataset links are included
        for link in all_links:
            if link not in final_llm_answer:
                final_llm_answer += f"\n\nYou can access the dataset here: {link}"
        
        return final_llm_answer.strip()

    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return f"Error generating response: {str(e)}"


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


def directly_use_llm_for_follow_up(query, refined_datasets, previous_answer, chatbot, data_df):
    """
    Improved follow-up function with dynamic SPARQL query generation.
    """
    # Step 1: Classify query type
    query_type = classify_query(query)

    # Step 2: Generate SPARQL query dynamically
    sparql_query = build_dynamic_sparql_query(query_type)

    # Step 3: Execute the query on the RDF graph
    sparql_results = query_rdf_graph(sparql_query)

    # Step 4: Format the SPARQL results
    graph_answer = ""
    if sparql_results:
        if query_type == "structure":
            graph_answer = "Dataset Structure:\n\n" + "\n".join(
                [f"Column: {row.label}, Data Type: {row.dataType}" for row in sparql_results]
            )
        elif query_type == "content":
            for row in sparql_results:
                dataset, title, summary, link, row_uri, property_uri, value = row
                graph_answer += (
                    f"Dataset Title: {title}\n"
                    f"Summary: {summary}\n"
                    f"Link: {link}\n"
                    f"Row: {row_uri}\n"
                    f"Property: {property_uri}\n"
                    f"Value: {value}\n\n"
                )
        else:
            graph_answer = "\n".join([f"{row.s}, {row.p}, {row.o}" for row in sparql_results])

    # Step 5: Create follow-up prompt for LLM
    follow_up_prompt = (
        f"Previously:\n{previous_answer}\n\n"
        f"Follow-up question: '{query}'\n"
        f"Latest graph data:\n{graph_answer}\n\n"
        "Provide an updated response."
    )

    try:
        # Step 6: Use the chatbot to generate a response
        follow_up_answer = chatbot.generate_response(context=previous_answer, query=follow_up_prompt)
        return follow_up_answer.strip()
    except Exception as e:
        logger.error(f"Error generating follow-up: {e}")
        return "Could not process the follow-up question. Please try again."
