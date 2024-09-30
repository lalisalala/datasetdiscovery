from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse  # We will use HTMLResponse for formatting
from pydantic import BaseModel
from streamline import run_streamline_process  # Import your existing streamline function
from fastapi.middleware.cors import CORSMiddleware  # Import for CORS handling

app = FastAPI()

# Enable CORS for all origins (you can restrict this to specific origins if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace '*' with a specific domain if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the structure of the input request
class QueryInput(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def chatbot_ui():
    """
    Serve a simple HTML page with a chatbot-like interface, adjusted to fill the entire browser window.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot</title>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
            }
            #chat-container {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                width: 100%;
                height: 100%;
                max-height: 100vh;
                background-color: white;
            }
            #chat-box {
                flex-grow: 1;
                padding: 15px;
                overflow-y: auto;
                border-bottom: 1px solid #ccc;
            }
            #chat-input-container {
                display: flex;
                padding: 10px;
                border-top: 1px solid #ccc;
                background-color: #fff;
            }
            #chat-input {
                flex-grow: 1;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-right: 10px;
            }
            #send-btn {
                padding: 10px;
                font-size: 16px;
                border-radius: 4px;
                border: none;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }
            #send-btn:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat-box"></div>
            <div id="chat-input-container">
                <input type="text" id="chat-input" placeholder="Ask your question..." />
                <button id="send-btn" onclick="sendQuery()">Send</button>
            </div>
        </div>

        <script>
            async function sendQuery() {
                const inputField = document.getElementById("chat-input");
                const query = inputField.value;
                inputField.value = ""; // Clear the input field
                if (query.trim() === "") return;

                // Display user query in chatbox
                const chatBox = document.getElementById("chat-box");
                chatBox.innerHTML += `<div><strong>You:</strong> ${query}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                // Send the query to the backend
                const response = await fetch("/run_analysis/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ query }),
                });

                // Ensure that the response is JSON and handled properly
                const result = await response.json();
                console.log(result); // Log the response for debugging purposes

                // Display chatbot response in chatbox, assuming it's properly formatted with <br> tags
                chatBox.innerHTML += `<div><strong>Chatbot:</strong> ${result.final_answer}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
            }
        </script>
    </body>
    </html>
    """


# Endpoint to handle the chatbot queries
@app.post("/run_analysis/", response_class=JSONResponse)
def run_analysis(query_input: QueryInput):
    """
    This endpoint runs the analysis using the 'run_streamline_process' function 
    and returns the final result formatted for HTML.
    """
    query = query_input.query  # Extract the query from the request
    final_answer = run_streamline_process(query)  # Call the streamline process with the query

    # Replace newlines with <br> tags to preserve formatting in the frontend
    formatted_answer = final_answer.replace("\n", "<br>")

    # Return the formatted answer as JSON
    return JSONResponse(content={"final_answer": formatted_answer})
