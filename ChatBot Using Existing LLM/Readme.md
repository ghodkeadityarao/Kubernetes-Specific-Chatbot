# Chat with Kubernetes Docs

This project enables users to interact with Kubernetes documentation in a conversational way. It uses Google's Gemini model for Q&A and a FAISS-based vector store for efficient similarity search over Kubernetes documentation.

## Features
- Scrapes Kubernetes documentation from official Kubernetes website.
- Breaks down the scraped content into chunks for embedding.
- Uses the FAISS vector store to index the content.
- Uses a custom Q&A model based on Google Generative AI (Gemini) to answer user queries.
- Streamlit frontend for interacting with the chatbot interface.

## Prerequisites
Before running the application, make sure you have the following:
- Python
- Gemini API Key
- Necessary Python libraries

## Setup and Installation

1. **Clone the Repository:**

    ```
    git clone <repository-url> cd <repository-name>
    ```

2. **Install Required Libraries:**
   Create a virtual environment (optional but recommended) and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
3. **Set Up Google API Key:**
- Sign up for Google AI Studio `https://aistudio.google.com/app/apikey` and generate the API key.
- Set your Google Cloud API Key by adding it to a `.env` file:
  
  ```
  GOOGLE_API_KEY=your-google-api-key-here
  ```

4. **Run the Streamlit App:**
    To start the chatbot application, use the following command:
    
    ```
    streamlit run app.py
    ```


## Code Explanation

### 1. **Scraping Kubernetes Documentation**
- The function `scrape_kubernetes_docs` scrapes multiple pages of Kubernetes documentation.
- The documentation is fetched from URLs like `https://kubernetes.io/docs/concepts/overview/` and several others.
- The HTML content of each page is parsed using `BeautifulSoup`, and the text is extracted and combined.

### 2. **Text Chunking**
- The `get_text_chunks` function splits the scraped documentation into smaller chunks.
- This is done to make it easier to process and index the text for similarity search.
- It uses `RecursiveCharacterTextSplitter` with a chunk size of 10,000 characters and 1,000-character overlap for better context retention.

### 3. **Creating and Saving Vector Store**
- The function `create_and_save_vector_store` creates a vector store using the `GoogleGenerativeAIEmbeddings` model for embedding the text.
- It uses the FAISS library to create a high-performance vector store for similarity search.
- The vector store is saved locally to the path specified in `INDEX_PATH`.

### 4. **Loading the Vector Store**
- The function `load_vector_store` loads the FAISS vector store if it already exists.
- If the vector store is not found, it returns `None`.

### 5. **Conversational Chain**
- The `get_conversational_chain` function defines the Q&A chain that uses Google’s Gemini-based model for responding to user queries.
- A custom prompt template is defined to ensure that responses are related to Kubernetes.
- The `load_qa_chain` function loads the question-answering chain using the Gemini model.

### 6. **Handling User Input**
- The `user_input` function takes the user query and performs a similarity search on the FAISS vector store.
- It retrieves the top 5 relevant documents and passes them to the conversational chain along with the user’s question.
- The response is then displayed on the Streamlit interface.

### 7. **Streamlit Interface**
- The `main` function sets up the Streamlit interface.
- It checks if the vector store is loaded. If not, it scrapes the Kubernetes docs, splits them into chunks, creates the vector store, and saves it.
- Users can enter their questions in the Streamlit app, and the system will return an appropriate response based on the content of the Kubernetes documentation.
