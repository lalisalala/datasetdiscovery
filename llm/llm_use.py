import logging
from typing import Any
import pandas as pd  
from llm.llm_chatbot import LLMChatbot
from sparql_query import retrieve_metadata, query_rdf_graph, classify_query, build_dynamic_sparql_query_optimized

logger = logging.getLogger(__name__)

def directly_use_llm_for_answer(metadata_list, query: str, chatbot: LLMChatbot, additional_context: str = "") -> str:
    """
    Use the LLM to analyze metadata and generate structured responses for users.
    Args:
        metadata_list (list): A list of metadata dictionaries for relevant datasets.
        query (str): The user's query.
        chatbot (LLMChatbot): Instance of the chatbot.
        additional_context (str): Additional information from the knowledge graph or SPARQL queries.

    Returns:
        str: LLM-generated response in a user-friendly format.
    """
    # Prepare metadata summary for LLM input
    metadata_summary = ""
    for idx, metadata in enumerate(metadata_list, start=1):
        # Format each dataset's metadata with key details
        metadata_summary += f"Dataset {idx}:\n"
        metadata_summary += f"  Title: {metadata.get('title', 'N/A')}\n"
        metadata_summary += f"  Summary: {metadata.get('summary', 'N/A')}\n"
        metadata_summary += f"  Publisher: {metadata.get('publisher', 'N/A')}\n"
        metadata_summary += f"  Topics: {metadata.get('topic', 'N/A')}\n"
        metadata_summary += f"  Format: {metadata.get('format', 'N/A')}\n"
        metadata_summary += f"  Link: {metadata.get('links', 'N/A')}\n\n"

    # Prepare the LLM prompt
    llm_prompt = (
        f"You are an intelligent assistant helping users find datasets.\n\n"
        f"User query: {query}\n\n"
        f"{'Additional context from the knowledge graph:\n' + additional_context if additional_context else ''}\n\n"
        "Here is the metadata for relevant datasets:\n"
        + metadata_summary
        + "\n\nPlease provide a clear and concise summary of the most relevant datasets that match the user's query."
    )

    # Log the prompt for debugging
    logger.debug(f"LLM Prompt:\n{llm_prompt}")

    try:
        # Use the LLM to generate a response
        final_answer = chatbot.generate_response(context=metadata_summary, query=llm_prompt)

        # Post-process the response to ensure it's user-friendly
        unified_output = "Here are the datasets matching your query:\n\n"
        for idx, metadata in enumerate(metadata_list, start=1):
            unified_output += f"Dataset {idx}:\n"
            unified_output += f"  Title: {metadata.get('title', 'N/A')}\n"
            unified_output += f"  Summary: {metadata.get('summary', 'N/A')}\n"
            unified_output += f"  Publisher: {metadata.get('publisher', 'N/A')}\n"
            unified_output += f"  Topics: {metadata.get('topic', 'N/A')}\n"
            unified_output += f"  Format: {metadata.get('format', 'N/A')}\n"
            unified_output += f"  Link: {metadata.get('links', 'N/A')}\n\n"

        # Combine the LLM response and formatted dataset details
        return unified_output.strip()

    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return f"Error generating response: {str(e)}"



def use_llm_for_metadata_selection(df, query: str, chatbot: LLMChatbot) -> pd.DataFrame:
    """
    Use the LLM to select relevant datasets based on metadata.

    Args:
        df (pd.DataFrame): Dataframe containing metadata (title, summary, links).
        query (str): The user query to determine relevant datasets.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        pd.DataFrame: A dataframe containing only the relevant datasets.
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


def directly_use_llm_for_follow_up(query, refined_datasets, previous_answer, chatbot):
    """
    Improved follow-up function using SPARQL results and metadata.

    Args:
        query (str): The follow-up query.
        refined_datasets (list): Metadata for the refined datasets.
        previous_answer (str): The previous LLM answer for context.
        chatbot (LLMChatbot): Instance of the chatbot.

    Returns:
        str: LLM-generated follow-up response.
    """
    # Step 1: Classify query type
    query_type = classify_query(query)

    # Step 2: Generate SPARQL query dynamically
    sparql_query = build_dynamic_sparql_query_optimized(query_type)

    # Step 3: Execute the query on the RDF graph
    sparql_results = query_rdf_graph(sparql_query)

    # Step 4: Format the SPARQL results
    graph_answer = ""
    if sparql_results:
        if query_type == "metadata":
            graph_answer = "Metadata Overview:\n\n" + "\n".join(
                [f"{row.dataset}: Title={row.title}, Summary={row.summary}, Link={row.link}" for row in sparql_results]
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
