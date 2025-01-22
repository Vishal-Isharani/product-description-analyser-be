from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from constants import SYSTEM_PROMPT, INSTRUCTIONS
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# create a retriever that is aware of the conversation history
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path

import os

load_dotenv()

env = os.getenv("ENV", "dev")

app = FastAPI(
    title="Product Ingredients Analyzer",
    description="Analyze the ingredients of a product",
    version="1.0.0",
)

origins = []

if env == "dev":
    origins.append("http://localhost:5173")
else:
    origins.append("https://ingredientsui.qodist.in")
    origins.append("https://localhost")
    origins.append("capacitor://localhost")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeURLRequest(BaseModel):
    url: str
    request_id: str


class AnalyzeResponse(BaseModel):
    Ingredients: list[str]
    HealthImplications: list[str]
    Considerations: list[str]
    NutritionInformation: list[str]
    Conclusion: str
    ProductName: str


class ChatRequest(BaseModel):
    message: str
    request_id: str


class ChatResponse(BaseModel):
    message: str


def get_agent():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Food Product Analyzer",
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        tools=[DuckDuckGo()],
        structured_outputs=True,
        response_model=AnalyzeResponse,
    )
    return agent


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/analyze")
async def analyze(file: UploadFile, request_id: str):
    print("request_id", request_id)
    temp_path = await save_to_temp_file(file)
    response = analyze_image(temp_path, request_id)
    os.unlink(temp_path)
    return response


@app.post("/analyze-url")
async def analyze_url(request: AnalyzeURLRequest):
    response = analyze_image(request.url, request.request_id)
    return response


@app.post("/chat")
async def chat(request: ChatRequest):
    vectorstore = load_vector_store(request.request_id)
    if not vectorstore:
        raise ValueError("Vector store not found. Please try again later.")

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.\
    Always return the answer in markdown format."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": request.message},
        config={
            "configurable": {"session_id": request.request_id},
        },
    )

    return response.get("answer", "")


async def save_to_temp_file(file):
    suffix = f".{file.content_type.split('/')[-1]}"
    with NamedTemporaryFile(dir=".", suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_file_path = tmp.name  # Save the temp file path
        return temp_file_path


def analyze_image(image_path, request_id):
    agent = get_agent()
    response = agent.run(
        "Analyze the product image",
        images=[image_path],
    )

    prepare_knowledge_base(response, request_id)

    return response.content


def prepare_knowledge_base(response: AnalyzeResponse, request_id: str):
    text = f"**Metadata**: \n\n"
    text += f"Product Name: {response.content.ProductName}\n"
    text += f"Ingredients: {', '.join(response.content.Ingredients)}\n"
    text += f"Health Implications: {', '.join(response.content.HealthImplications)}\n"
    text += f"Considerations: {', '.join(response.content.Considerations)}\n"
    text += (
        f"Nutrition Information: {', '.join(response.content.NutritionInformation)}\n"
    )
    text += f"Conclusion: {response.content.Conclusion}\n\n"

    # text += f"**Chat History**: \n\n"

    create_vector_store(text, request_id)

    return text


def create_vector_store(text, request_id):
    """Create and save vector store"""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(texts, embeddings)

    # Save vector store
    vector_store.save_local(f"vector_stores/{request_id}")

    # Save to cloud storage
    # _save_vector_store_cloud(request_id)

    return vector_store


# def _save_vector_store_cloud(request_id):
#     """Save vector store to cloud storage"""
#     import os
#     from pathlib import Path

#     local_directory = Path(f"vector_stores/{request_id}")

#     for root, dirs, files in os.walk(local_directory):
#         for file in files:
#             local_file_path = os.path.join(root, file)
#             local_file = Path(local_file_path).resolve()

#             try:
#                 storage_service.upload_file_from_path(
#                     local_file, f"{request_id}/{file}"
#                 )
#             except Exception as e:
#                 print(f"Error uploading file to cloud storage: {e}")


def load_vector_store(request_id):
    """Load vector store from local or cloud storage"""
    storage_dir = Path("vector_stores")
    file_path = storage_dir / request_id
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76
    )

    if file_path.exists():
        print("Loading vector store from local")
        vector_store = FAISS.load_local(
            str(file_path), embeddings, allow_dangerous_deserialization=True
        )
        return vector_store
    else:
        print("Loading vector store from cloud")
        return None
        # return self._load_vector_store_cloud(document_id)


# def _load_vector_store_cloud(self, document_id):
#     """Load vector store from cloud storage"""
#     # Create local directory if it doesn't exist
#     storage_dir = Path("vector_stores")
#     local_file_path = storage_dir / document_id
#     local_file_path.mkdir(exist_ok=True, parents=True)

#     try:
#         # Download faiss file
#         download_faiss_file = self.storage_service.download_file(
#             f"{document_id}/index.faiss"
#         )
#         # Save faiss file to local directory
#         download_faiss_file.save_to(
#             str(local_file_path / "index.faiss"), mode="wb+", allow_seeking=False
#         )

#         # Download pkl file
#         download_pkl_file = self.storage_service.download_file(
#             f"{document_id}/index.pkl"
#         )
#         # Save pkl file to local directory
#         download_pkl_file.save_to(
#             str(local_file_path / "index.pkl"), mode="wb+", allow_seeking=False
#         )

#         # Load vector store from local directory
#         if local_file_path.exists():
#             embeddings = OpenAIEmbeddings()
#             return FAISS.load_local(
#                 str(local_file_path),
#                 embeddings,
#                 allow_dangerous_deserialization=True,
#             )
#     except Exception as e:
#         print(f"Error loading vector store from cloud: {e}")
#         return None
