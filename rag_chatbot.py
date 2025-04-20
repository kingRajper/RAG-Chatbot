import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, book_dir="books", db_dir="db"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not defined. Please check environment variables.")

        self.book_dir = book_dir
        self.persist_directory = os.path.join(db_dir, "chroma_db_with_metadata")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)

        self.db = self._load_or_create_vector_store()
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.pipeline = self._build_pipeline()

    def _load_or_create_vector_store(self):
        if not os.path.exists(self.book_dir):
            raise FileNotFoundError(f"The directory {self.book_dir} does not exist.")

        if not os.path.exists(self.persist_directory):
            logger.info("Vector store not found. Creating a new one...")
            documents = self._load_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Number of chunks created: {len(chunks)}")
            db = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_directory)
            db.persist()
            logger.info("Vector store created and persisted successfully.")
        else:
            logger.info("Loading existing vector store.")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return db

    def _load_documents(self):
        book_files = [f for f in os.listdir(self.book_dir) if f.endswith(".txt")]
        documents = []
        for book_file in book_files:
            file_path = os.path.join(self.book_dir, book_file)
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                doc = loader.load()[0]
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="ISO-8859-1")
                doc = loader.load()[0]
            doc.metadata = {"source": book_file}
            documents.append(doc)
        if not documents:
            raise ValueError("No documents loaded successfully.")
        return documents

    def _build_pipeline(self):
        # Reformulation prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
                       "formulate a standalone question that can be understood without the chat history. "
                       "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = RunnableSequence(
            {
                "reformulated_query": contextualize_prompt | self.llm | StrOutputParser(),
                "chat_history": lambda x: x["chat_history"],
                "input": lambda x: x["input"],
            } | RunnablePassthrough.assign(
                documents=lambda x: self.retriever.invoke(x["reformulated_query"])
            )
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. "
                       "Use the following pieces of retrieved context to answer the question. "
                       "If you don't know the answer, just say you don't know. "
                       "Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        return RunnableSequence(
            history_aware_retriever,
            {
                "context": lambda x: "\n".join([
                    f"[From {doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
                    for doc in x["documents"]
                ]),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"]
            } | qa_prompt | self.llm | StrOutputParser()
        )

    def chat(self):
        logger.info("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history = []
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            try:
                response = self.pipeline.invoke({
                    "input": query,
                    "chat_history": chat_history
                })
                print(f"AI: {response}")
                chat_history.append(HumanMessage(content=query))
                chat_history.append(SystemMessage(content=response))
            except Exception as e:
                logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.chat()
