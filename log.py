from datetime import datetime

def log_execution_details(query, relevant_datasets, final_answer, successful_links, total_time):
    """
    Log the details of the execution, including query, relevant datasets, final answer,
    successfully downloaded links, and execution time.
    """
    with open('execution_log.txt', 'a') as log_file:
        log_file.write(f"Execution Timestamp: {datetime.now()}\n")
        log_file.write(f"User Query: {query}\n\n")
        
        log_file.write("Relevant Datasets:\n")
        log_file.write(relevant_datasets[['title', 'links']].to_string(index=False))
        log_file.write("\n\n")

        log_file.write(f"Successfully Downloaded Links:\n")
        for link in successful_links:
            log_file.write(f"{link}\n")
        
        log_file.write(f"\nTotal Time Taken: {total_time:.2f} seconds\n")
        log_file.write(f"LLM Output:\n{final_answer}\n\n")
        log_file.write("=" * 80 + "\n\n")
