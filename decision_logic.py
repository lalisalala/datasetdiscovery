def decide_faiss_or_llm(query, data_df, chatbot):
    """
    Use the LLM to decide whether the second FAISS search is necessary or whether 
    the LLM can directly answer the query using data from the dataset.

    Args:
        query (str): The user query.
        data_df (pd.DataFrame): The dataframe containing the downloaded data.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        dict: A dictionary indicating whether FAISS is needed, and if so, how many results (k).
              Example: {'use_faiss': True, 'k': 5}.
    """
    # Provide a smaller sample of the dataset to the LLM
    sample_data = data_df.head(10).to_string()

    # Self-confident prompt to encourage the LLM to take control of the analysis
    prompt = (
        f"The user query is: '{query}'.\n\n"
        f"Here is a sample of the dataset (first 10 rows):\n{sample_data}\n\n"
        "Considering your ability to understand and analyze this dataset directly, can you handle this query "
        "on your own without FAISS search? "
        "Respond with 'LLM: yes' if you can handle it directly or 'FAISS: yes' if a deeper search is needed. "
        "If FAISS is needed, also provide the number of top results (k) in this format: 'k: <number>'."
    )

    # Send the prompt to the LLM
    decision = chatbot.generate_response(context=sample_data, query=prompt)

    # Parse the LLM's decision to favor LLM responses
    if 'LLM: yes' in decision:
        return {'use_faiss': False}
    elif 'FAISS: yes' in decision:
        try:
            k_value = int(decision.split('k: ')[1].strip())
            return {'use_faiss': True, 'k': max(1, k_value)}
        except (ValueError, IndexError):
            return {'use_faiss': True, 'k': 5}  # Default to 5 if parsing fails
    else:
        return {'use_faiss': False}  # Favor LLM if the response isn't clear



def decide_faiss_or_llm_for_metadata(query, df_with_summaries, chatbot):
    """
    Use the LLM to decide whether a FAISS search is necessary for the metadata or if the LLM 
    can directly parse through datasets.csv and select relevant datasets.

    Args:
        query (str): The user's query.
        df_with_summaries (pd.DataFrame): The dataframe containing metadata summaries and links.
        chatbot (LLMChatbot): An instance of the LLMChatbot class.

    Returns:
        dict: A dictionary indicating whether FAISS is needed, and if so, how many results (k).
              Example: {'use_faiss': True, 'k': 5} or {'use_faiss': False}.
    """
    # Provide a smaller sample of the metadata to the LLM
    sample_metadata = df_with_summaries.head(3).to_string()

    # More self-confident prompt to reduce reliance on FAISS
    prompt = (
        f"The user query is: '{query}'.\n\n"
        f"Here is a small sample of the dataset metadata:\n{sample_metadata}\n\n"
        "Considering your ability to understand and analyze text directly, can you handle this query on your own "
        "without needing FAISS search for more relevance checking? "
        "Respond with 'LLM: yes' if you can handle it directly or 'FAISS: yes' if a deeper search is needed. "
        "If FAISS is needed, also provide the number of top results (k) in this format: 'k: <number>'."
    )

    # Send the prompt to the LLM
    decision = chatbot.generate_response(context=sample_metadata, query=prompt)

    # Parse the LLM's decision to favor LLM responses
    if 'LLM: yes' in decision:
        return {'use_faiss': False}
    elif 'FAISS: yes' in decision:
        try:
            k_value = int(decision.split('k: ')[1].strip())
            return {'use_faiss': True, 'k': max(1, k_value)}
        except (ValueError, IndexError):
            return {'use_faiss': True, 'k': 5}  # Default to 5 if parsing fails
    else:
        return {'use_faiss': False}  # Favor LLM if the response isn't clear


def identify_text_columns(df):
    """
    Identify columns in the dataframe that contain text (string-based columns).
    
    Args:
        df (pd.DataFrame): The dataframe containing the downloaded data (data.csv).
        
    Returns:
        list: A list of column names that are text-based.
    """
    text_columns = []
    for col in df.columns:
        # Check if the column contains object data type (usually string/text data)
        if df[col].dtype == 'object':
            text_columns.append(col)
    return text_columns