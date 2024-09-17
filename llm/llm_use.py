def directly_use_llm_for_answer(data_df, query, chatbot):
    """
    Use the LLM to directly analyze data from data.csv and answer the query.
    
    Args:
        data_df (pd.DataFrame): The dataframe containing the downloaded data.
        query (str): The user query to answer.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.
    
    Returns:
        str: The LLM's answer based on the data and the query.
    """
    # Convert the dataframe to a string to send to the LLM (this can be refined)
    # Limit the amount of data sent to avoid overwhelming the LLM
    data_string = data_df.head(100).to_string()  # Take only the first 100 rows for now

    # Construct the LLM prompt
    prompt = (
        f"The user query is: '{query}'.\n\n"
        f"Here is a sample of the dataset (first 100 rows):\n{data_string}\n\n"
        "Based on the dataset, please provide a detailed and accurate answer to the user's query."
    )
    
    # Send the prompt to the LLM and get the answer
    final_answer = chatbot.generate_response(context=data_string, query=query)
    return final_answer

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
