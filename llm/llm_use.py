def directly_use_llm_for_answer(data_df, query, chatbot, chunk_size=100):
    """
    Use the LLM to directly analyze data from data.csv and answer the query in chunks if the dataset is large.

    Args:
        data_df (pd.DataFrame): The dataframe containing the downloaded data.
        query (str): The user query to answer.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.
        chunk_size (int): Number of rows to process at a time in chunks.

    Returns:
        str: The LLM's comprehensive answer based on the data and the query.
    """
    final_answer = ""
    
    # Process the dataset in chunks
    for i in range(0, len(data_df), chunk_size):
        chunk = data_df.iloc[i:i + chunk_size].to_string()

        # Construct the LLM prompt
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Here is a sample of the dataset (rows {i} to {i + chunk_size}):\n{chunk}\n\n"
            "Based on this data, provide a detailed analysis and answer to the user's query."
        )

        # Send the prompt to the LLM and accumulate the answers
        chunk_answer = chatbot.generate_response(context=chunk, query=prompt)
        final_answer += chunk_answer + "\n\n"  # Accumulate the answers

    return final_answer.strip()  # Return the complete answer



def use_llm_for_metadata_selection(df, query, chatbot):
    """
    Use the LLM to directly parse through the metadata summaries in datasets.csv and select the relevant datasets.

    Args:
        df (pd.DataFrame): Dataframe containing metadata (title, summary, links).
        query (str): The user query to determine relevant datasets.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        pd.DataFrame: A dataframe containing only the relevant datasets based on the LLM's decision.
    """
    relevant_indices = []

    # Iterate through the metadata summaries and use the LLM to determine relevance
    for idx, row in df.iterrows():
        metadata_content = f"Title: {row['title']}\nSummary: {row['summary']}\nLink: {row['links']}"
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Below is a dataset metadata entry:\n\n{metadata_content}\n\n"
            "Is this dataset relevant to the query? Answer with 'yes' or 'no'."
        )

        # Ask the LLM if the dataset is relevant
        llm_response = chatbot.generate_response(context=metadata_content, query=prompt)
        
        if 'yes' in llm_response.lower():
            relevant_indices.append(idx)

    # Filter the dataframe to include only relevant datasets
    relevant_datasets = df.iloc[relevant_indices]
    return relevant_datasets

def refine_llm_on_relevant_datasets(relevant_datasets, chatbot, query):
    """
    Use the LLM to further refine or narrow down the relevant datasets based on the query.
    
    Args:
        relevant_datasets (pd.DataFrame): DataFrame containing the datasets identified as relevant by FAISS.
        chatbot (LLMChatbot): An instance of the LLMChatbot class to interact with the LLM.
        query (str): The original query to refine against.
        
    Returns:
        pd.DataFrame: A more refined set of relevant datasets after LLM filtering.
    """
    refined_indices = []

    # Loop over each dataset and ask the LLM to determine its relevance
    for idx, row in relevant_datasets.iterrows():
        # Construct the metadata/content context for the LLM, including year and other relevant fields
        metadata_content = f"Title: {row['title']}\n"
        if 'metadatasummary' in row:
            metadata_content += f"Metadata Summary: {row['metadatasummary']}\n"
        if 'content' in row:
            metadata_content += f"Content: {row['content'][:500]}..."  # Use a sample of the content (first 500 characters)
        
        # Add query details to make the LLM aware of specific aspects to focus on (e.g., year, category, etc.)
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Here is some information about the dataset:\n{metadata_content}\n\n"
            "Please determine if this dataset is highly relevant to the query. Pay special attention to the year, category, "
            "and any specific keywords from the query, such as '{query}'. If the dataset is highly relevant, "
            "answer with 'yes', and if not, answer 'no'. Provide a reason for your decision."
        )

        # Get the LLM response
        llm_response = chatbot.generate_response(context=metadata_content, query=prompt)

        # If the LLM says it's relevant, keep track of it
        if 'yes' in llm_response.lower():
            refined_indices.append(idx)

    # If LLM doesn't find any relevant datasets, return the original FAISS results
    if not refined_indices:
        print("LLM found no highly relevant datasets. Proceeding with the FAISS-selected datasets.")
        return relevant_datasets  # Return the original FAISS-selected datasets

    # Ensure valid indices and avoid IndexError
    valid_indices = [i for i in refined_indices if i < len(relevant_datasets)]

    # Filter the DataFrame to include only the refined datasets
    refined_datasets = relevant_datasets.iloc[valid_indices]
    
    print(f"LLM refined the relevant datasets down to {len(refined_datasets)} datasets.")
    
    return refined_datasets
