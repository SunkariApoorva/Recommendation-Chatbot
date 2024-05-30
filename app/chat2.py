import os
import json
import fitz  # PyMuPDF
import time
import numpy as np
import asyncio
import aiofiles
import logging
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import chainlit as cl
from typing import Iterator, List
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langdetect import detect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_folder = "C:/Users/sunka/RecommendationSystem/data"
embedding_folder = "C:/Users/sunka/RecommendationSystem/embeddings"
embedding_file = os.path.join(embedding_folder, "embeddings.json")
chroma_db_file = os.path.join(embedding_folder, "chroma_db.db")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
ANTHROPIC_API_KEY = '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'

Base = declarative_base()

class APIStatusError(Exception):
    pass

class CustomRetriever:
    def __init__(self, embeddings_with_names):
        self.embeddings_with_names = embeddings_with_names

    def _get_relevant_documents(self, query, num_documents=5):
        query_embedding = embedding_model.encode([query])[0]
        distances = [(item['file_name'], np.linalg.norm(query_embedding - np.array(item['embedding'])))
                     for item in self.embeddings_with_names]
        distances.sort(key=lambda x: x[1])
        return [doc[0] for doc in distances[:num_documents]]

class DocumentModel(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    embedding = Column(Text)
    document_metadata = Column(Text)

def initialize_database():
    engine = create_engine(f'sqlite:///{chroma_db_file}', echo=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session

engine, session = initialize_database()

class CustomDocumentLoader:
    def __init__(self, documents: list, chunk_size: int):
        self.documents = documents
        self.chunk_size = chunk_size
        logger.info("Initializing CustomDocumentLoader with %d documents", len(documents))

    def lazy_load(self) -> Iterator[Document]:
        for file_name, text in self.documents:
            if not isinstance(text, str):
                text = str(text)  # Convert to string if it's not already
            for i in range(0, len(text), self.chunk_size):
                page_content = text[i:i + self.chunk_size]
                yield Document(page_content=page_content, metadata={"source": file_name})

async def process_pdf_and_embed(pdf_file):
    logger.info(f"Processing PDF: {pdf_file}")
    text = extract_text_from_pdf(pdf_file)
    embedding = embedding_model.encode([text])[0]
    return {'file_name': pdf_file, 'text': text, 'embedding': embedding.tolist()}

async def initialize_chroma_database(session, embeddings_with_names):
    try:
        existing_documents = session.query(DocumentModel).all()
        logger.info(f"Number of existing documents: {len(existing_documents)}")

        new_embeddings = {item['file_name']: item for item in embeddings_with_names}

        if session.query(DocumentModel).count() == 0:
            logger.info("Chroma database is empty. Creating new Chroma database...")
            all_embeddings = [item['embedding'] for item in embeddings_with_names]
            all_document_names = [item['file_name'] for item in embeddings_with_names]
            document_chunks = [Document(page_content=json.dumps(emb)) for emb in all_embeddings]
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            chroma_db = Chroma.from_documents(document_chunks, embedding_function)

            for name, embedding in zip(all_document_names, all_embeddings):
                document = DocumentModel(embedding=json.dumps(embedding), document_metadata=name)
                session.add(document)
            session.commit()
            logger.info("Chroma database created and initialized with embeddings.")
        else:
            logger.info("Chroma database already exists. Checking for new embeddings...")

            new_document_names = set(new_embeddings.keys())
            existing_document_names = {doc.document_metadata for doc in existing_documents}
            documents_to_insert = [(name, new_embeddings[name]['embedding']) for name in new_document_names - existing_document_names]

            if documents_to_insert:
                logger.info(f"Inserting {len(documents_to_insert)} new documents into the database...")
                chunk_loader = CustomDocumentLoader(documents=documents_to_insert, chunk_size=1000)
                document_chunks = list(chunk_loader.lazy_load())
                embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                chroma_db = Chroma.from_documents(document_chunks, embedding_function)

                for name, embedding in documents_to_insert:
                    document = DocumentModel(embedding=json.dumps(embedding), document_metadata=name)
                    session.add(document)
                session.commit()
                logger.info("Chroma database updated with new embeddings.")
            else:
                logger.info("No new embeddings found. Using existing Chroma database.")
                existing_embeddings = [json.loads(doc.embedding) for doc in existing_documents]
                document_chunks = [Document(page_content=json.dumps(emb)) for emb in existing_embeddings]
                embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                chroma_db = Chroma.from_documents(document_chunks, embedding_function)

        return chroma_db

    except Exception as e:
        logger.error(f"Error initializing Chroma database: {e}")
        return None

async def load_embeddings():
    async with aiofiles.open(embedding_file, "r") as f:
        return json.loads(await f.read())

async def save_embeddings(embeddings_with_names):
    async with aiofiles.open(embedding_file, "w") as f:
        await f.write(json.dumps(embeddings_with_names))

async def process_pdf_files(pdf_files, batch_size=5):
    start_time = time.time()
    embeddings_with_names = []
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        tasks = [process_pdf_and_embed(pdf_file) for pdf_file in batch]
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            result['file_name'] = result['file_name'].strip('"')
            embeddings_with_names.extend(batch_results)
    duration = time.time() - start_time
    logger.info(f"Processed {len(pdf_files)} PDFs in {duration:.2f} seconds")
    return embeddings_with_names

async def update_embeddings():
    if os.path.exists(embedding_file):
        logger.info("Loading existing embeddings...")
        embeddings_with_names = await load_embeddings()
        logger.info(f"Loaded {len(embeddings_with_names)} embeddings.")
    else:
        logger.info("Embedding file does not exist. Creating new embeddings...")
        pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".pdf")]
        embeddings_with_names = await process_pdf_files(pdf_files, batch_size=5)
        await save_embeddings(embeddings_with_names)
    return embeddings_with_names

async def initialize_chat_anthropic():
    try:
        chat_anthropic = ChatAnthropic(
            model='claude-3-opus-20240229',
            temperature=0,
            streaming=True,
            anthropic_api_key=ANTHROPIC_API_KEY,
        )
        return chat_anthropic
    except Exception as e:
        logger.error(f"Error initializing ChatAnthropic: {e}")
        return None

async def start_chat(engine, session):
    logger.info("Starting chat setup...")
    start_time = time.time()

    embeddings_with_names = await update_embeddings()
    chroma_task = asyncio.create_task(initialize_chroma_database(session, embeddings_with_names))
    chroma_db = await chroma_task
    chat_anthropic = await initialize_chat_anthropic()

    duration = time.time() - start_time
    logger.info(f"Chroma database and chat_anthropic setup completed in {duration:.2f} seconds")

    if chroma_db and chat_anthropic:
        retriever = chroma_db.as_retriever()
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_anthropic,
            retriever=retriever,
            memory=memory,
        )

        cl.user_session.set("chain", conversational_chain)
        cl.user_session.set("embedding_model", embedding_model)

        logger.info("Chat setup completed.")
    else:
        logger.error("Chroma DB or ChatAnthropic initialization failed. Chat setup failed.")

@cl.on_chat_start
async def start():
    await cl.Avatar(name="Chatbot", url="/public/avatar.png").send()
    await start_chat(engine, session)

# async def api_call_with_retry(chain, message_content, relevant_docs, cb):
#     max_retries = 3
#     retry_delay = 5
#     for attempt in range(max_retries):
#         try:
#             logger.info(f"Calling the API (attempt {attempt + 1}/{max_retries})...")
#             # Refine the query structure to provide more context
#             query = construct_query(message_content, relevant_docs)
#             response = await chain.ainvoke(query, callbacks=[cb])
#             print("response:",response)
#             if response and 'answer' in response:
#                 return response['answer']
#         except asyncio.TimeoutError:
#             logger.warning("API call timed out.")
#         except Exception as e:
#             logger.error(f"Error during API call: {e}")

#         if attempt < max_retries - 1:
#             logger.info(f"Retrying in {retry_delay} seconds...")
#             await asyncio.sleep(retry_delay)
#     return None

async def api_call_with_retry(chain, message_content,relevant_docs, cb):
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling the API (attempt {attempt + 1}/{max_retries})...")
            # Refine the query structure to provide more context
            query = construct_query(message_content,relevant_docs)
            response = await chain.ainvoke(query, callbacks=[cb])
            if response and 'answer' in response:
                relevant_docs = retrieve_documents(response['answer'])  # Extract relevant documents from the API response
                return response  # Return the entire API response dictionary
        except asyncio.TimeoutError:
            logger.warning("API call timed out.")
        except Exception as e:
            logger.error(f"Error during API call: {e}")

        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
    return None  # Return None if all retries fail


def construct_query(message_content, relevant_docs):
    # Refine the query structure to provide more context or refine the query parameters
    # Here, we concatenate the user's message with relevant document titles to provide more context
    query = f"{message_content} Relevant documents: {' '.join(relevant_docs)}"
    return query


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message: {message.content}")
    try:
        chain = cl.user_session.get("chain")
        embedding_model = cl.user_session.get("embedding_model")
    except KeyError as e:
        logger.error(f"KeyError: {e} - The session may have been disconnected.")
        return

    cb = cl.AsyncLangchainCallbackHandler()

    try:
        async with asyncio.timeout(60):
            relevant_docs_initial = retrieve_documents(message.content)

            response = await api_call_with_retry(chain, message.content, relevant_docs_initial,cb)

            if response:
                logger.info("Received response from API")
                answer = response.get('answer', '')  # Extract the answer from the response dictionary
                relevant_docs_api = retrieve_documents(answer)  # Extract relevant documents from the API response

                # Combine initial relevant documents and new relevant documents from API response
                # Strip off quotes and remove duplicates
                relevant_docs_initial = [doc.strip('"') for doc in relevant_docs_initial]
                relevant_docs_api = [doc.strip('"') for doc in relevant_docs_api]
                all_relevant_docs = list(set(relevant_docs_initial + relevant_docs_api))

                if all_relevant_docs:
                    policy_document_info = "\n".join([f"- {doc}" for doc in all_relevant_docs])
                    response = f"Las siguientes políticas cubren su requerimiento:\n{policy_document_info}\n\nRespuesta del usuario: {answer}"
                else:
                    response = answer

                language = detect(message.content)
                if language == "es":
                    response = f"Las siguientes políticas cubren su requerimiento:\n{policy_document_info}\n\nRespuesta del usuario: {answer}"

                await cl.Message(content=response).send()
                logger.info(f"Sent response: {response}")
            else:
                await cl.Message(content="Failed to retrieve response from API. Please try again later.").send()
                logger.error("Failed to retrieve response from API.")

    except asyncio.TimeoutError:
        await cl.Message(content="The API call timed out. Please try again later.").send()
        logger.error("API call timed out.")
    except APIStatusError as e:
        await cl.Message(content="The service is currently overloaded. Please try again later.").send()
        logger.error(f"APIStatusError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await cl.Message(content="An unexpected error occurred. Please try again later.").send()


def retrieve_documents(query, top_k=5):
    logger.info(f"Retrieving documents for query: {query}")
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = nbrs.kneighbors([query_embedding], n_neighbors=top_k)
    relevant_docs = [sources[idx] for idx in indices[0]]
    logger.info(f"Relevant documents retrieved: {relevant_docs}")
    return relevant_docs

def find_closest_embeddings(response_embedding, top_k=5):
    distances, indices = nbrs.kneighbors([response_embedding], n_neighbors=top_k)
    closest_texts = [sources[idx] for idx in indices[0]]
    return closest_texts

def load_embeddings_from_db(session):
    documents = session.query(DocumentModel).all()
    embeddings = []
    sources = []
    for doc in documents:
        embedding = np.array(json.loads(doc.embedding))
        source = doc.document_metadata
        sources.append(source)
        embeddings.append(embedding)
    return np.array(embeddings), sources

embeddings, sources = load_embeddings_from_db(session)
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)


if __name__ == "__main__":
    cl.run()
