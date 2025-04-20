<h1> # RAG-Chatbot</h1>
<h3>Description </h3>
RAG-Chatbot is an advanced conversational AI system that combines Retrieval-Augmented Generation (RAG) with general language model capabilities to provide precise answers to user queries. It first searches for relevant information in a vector store (Chroma) containing processed documents. If the relevant context is found, the chatbot retrieves it and provides a context-aware response. If no relevant information is found, it defaults to generating a general response, just like any typical large language model (LLM) such as GPT-4.

<h3> Features </h3>

* __Context-Aware Retrieval__ First, it attempts to retrieve the answer from a vector store using context from previously processed documents.

* __Fallback to General LLM:__ If the context isn't available in the vector store, the chatbot generates an answer using GPT-4, similar to standard LLMs.

* __Document Processing:__ Loads and processes text documents into a vector store for quick retrieval.

* __Conversational Interface:__ Provides an interactive chat experience through a simple command-line interface.

* __Persistent Vector Store:__ Stores document embeddings and metadata to avoid repetitive processing and enable efficient querying.

<h3> Requirements</h3>
*Python 3.7+
*Libraries:
*langchain-core
*langchain-openai
*langchain-chroma
*langhcain-community
*dotenv
*openai
*logging

<h3>Installation</h3>
1 __Install the required dependencies__
  * pip install -r requirements.txt
2 __Set up environment variables__
  * Create a .env file in the root directory with your OpenAI API key:
  * OPENAI_API_KEY=your-api-key
3 __Ensure the following directories exist__
 * books/ - This is where the .txt documents should be stored.
 * db/ - This will hold the Chroma vector store database.
<h3>How to Use </h3>
1 __Run the Chatbot:__ Once everything is set up, run the RAGChatbot to start a conversation with the AI:
 * python rag_chatbot.py
2 __Interact with the AI:__
 * You will be prompted to type your query.
 * The chatbot first looks for the answer in the vector store. If it finds relevant context, it retrieves it and responds with an informed answer.
 * If no relevant information is found, the chatbot defaults to a general answer, using its general knowledge like ChatGPT.
 * Type 'exit' to end the conversation.

<h3> How It Works </h3>

1 __Document Processing:__
 * The chatbot processes and stores documents in the books/ directory into a vector store (Chroma) by splitting them into chunks and embedding them with OpenAI embeddings.
   
2 __Query Handling:__

 * Search in Vector Store: When a user asks a question, the chatbot first attempts to find relevant information from the vector store.

 * Fallback to LLM: If no relevant information is found in the vector store, the chatbot generates an answer using OpenAI's GPT-4 model as a fallback.

3 __Conversational RAG Pipeline:__

 * The chatbot uses LangChain to build a pipeline that reformulates the question, retrieves relevant context from the vector store, and combines it with the GPT-4 model to give the best response.



