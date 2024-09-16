import os
import pandas as pd
from search.metadata_search import query_faiss_index, generate_summaries_for_datasets
from search.faiss_index import create_faiss_index
from search.data_search import download_datasets, load_query_from_yaml
from llm.llm_chatbot import LLMChatbot
from llm.llm_use import directly_use_llm_for_answer, use_llm_for_metadata_selection
from search.data_analysis import create_faiss_index_for_data, search_data
import subprocess
from decision_logic import decide_faiss_or_llm, decide_faiss_or_llm_for_metadata, identify_text_columns
from web.webscraping import run_webscraping
import yaml 

def main():
    # Load the user query from the YAML file
    query = load_query_from_yaml()

    # Define the CSV file for saving the datasets
    csv_file = 'datasets.csv'

    # Always run web scraping to overwrite datasets.csv
    run_webscraping()

    # Load the dataset metadata from the CSV file
    df = pd.read_csv(csv_file)

    # Initialize the LLM chatbot for generating summaries and relevance checking
    chatbot = LLMChatbot(model_name='mistral')

    # Generate summaries using the LLM for all datasets (if not already done)
    print("Generating summaries using the LLM...")
    df_with_summaries = generate_summaries_for_datasets(df, chatbot)
    df_with_summaries[['title', 'metadatasummary', 'links']].to_csv(csv_file, index=False)

    # Ask the LLM to decide whether a FAISS search is needed for metadata
    print("Using LLM to decide if FAISS is necessary for the first search...")
    faiss_decision = decide_faiss_or_llm_for_metadata(query, df_with_summaries, chatbot)
    
    if not faiss_decision['use_faiss']:
        # If FAISS is not needed, directly use the LLM to select relevant datasets
        print("LLM decided FAISS is not necessary. Selecting datasets directly from metadata...")
        relevant_datasets = use_llm_for_metadata_selection(df_with_summaries, query, chatbot)
        print(f"Relevant datasets selected by the LLM:\n{relevant_datasets[['title', 'links']]}")

        # Download the relevant datasets using the links
        combined_df = download_datasets(relevant_datasets)

        # Save the datasets to data.csv with the generated LLM summary as a header
        if combined_df is not None:
            save_data_with_llm_metadata_header(relevant_datasets, combined_df, output_file='data.csv')

    else:
        # Use the LLM to dynamically determine the number of top results (k)
        print("LLM deciding how many results (k) to return for FAISS metadata search...")
        k = chatbot.determine_k_from_query(query)

        # Perform FAISS search based on the LLM's decision and determined k
        print(f"LLM decided FAISS is necessary. Performing FAISS search with top {k} results...")
        combined_text = df_with_summaries['metadatasummary'] + " " + df_with_summaries['links']
        model, metadata_index = create_faiss_index(combined_text.tolist())
        best_indices, best_distances = query_faiss_index(query, model, metadata_index, k=k)

        # Ensure indices are within bounds of the DataFrame
        valid_indices = [i for i in best_indices if i < len(df_with_summaries)]
        if not valid_indices:
            print("No valid indices found. Exiting.")
            return

        # Retrieve relevant datasets based on valid FAISS indices
        relevant_datasets = df_with_summaries.iloc[valid_indices]
        print(f"Best Metadata Results (Distances: {best_distances}):\n")
        print(relevant_datasets[['title', 'metadatasummary', 'links']])

        # Download the relevant datasets using the links
        combined_df = download_datasets(relevant_datasets)

        # Save the datasets to data.csv with the generated LLM summary as a header
        if combined_df is not None:
            save_data_with_llm_metadata_header(relevant_datasets, combined_df, output_file='data.csv')

    # Load the downloaded datasets (data.csv)
    data_df = pd.read_csv('data.csv')

    # Ask the LLM to decide whether a second FAISS search is needed on the downloaded data
    print("Using LLM to decide if a second FAISS search is necessary on the downloaded data...")
    second_faiss_decision = decide_faiss_or_llm(query, data_df, chatbot)

    if second_faiss_decision['use_faiss']:
        # Use the LLM to determine the number of results (k) for the data FAISS search
        print("LLM deciding how many results (k) to return for FAISS data search...")
        k = chatbot.determine_k_from_query(query)

        # Identify text columns for FAISS indexing
        text_columns = identify_text_columns(data_df)
        if not text_columns:
            print("No text columns found in the downloaded data.")
            return
        
        # Concatenate all text columns into a single string for each row
        data_df['combined_text'] = data_df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Create a FAISS index for the content in data.csv (second FAISS search)
        data_model, data_index = create_faiss_index_for_data(data_df['combined_text'].tolist())
        
        # Updated line, passing data_df['combined_text'] as the data argument
        data_results = search_data(query, data_model, data_index, data_df['combined_text'].tolist(), k=k)

        # Display the top data results
        for i, result in enumerate(data_results):
            print(f"Data Result {i+1}:\n{result}\n")
        
        # Use the LLM to generate a final answer using both FAISS search results
        final_answer = chatbot.generate_response(
            context=f"Metadata Results:\n{relevant_datasets[['title', 'metadatasummary', 'links']].to_string(index=False)}\n\n"
                    f"Data Results:\n{'\n'.join(data_results)}\n\n"
                    "Based on the metadata and data analysis, provide a detailed answer to the query.",
            query=query  # Passing the user query here
        )

    else:
        # If FAISS is not needed, directly use the LLM to analyze data and provide the answer
        print("LLM decided FAISS is not necessary for the downloaded data. Directly analyzing with LLM...")
        final_answer = directly_use_llm_for_answer(data_df, query, chatbot)

    print(f"Final Answer:\n{final_answer}")