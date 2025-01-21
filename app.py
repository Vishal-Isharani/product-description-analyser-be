from dotenv import load_dotenv
from phi.agent import Agent
from phi.knowledge.text import TextKnowledgeBase
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from constants import SYSTEM_PROMPT, INSTRUCTIONS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from phi.storage.agent.mongodb import MongoAgentStorage
from phi.vectordb.mongodb import MongoDBVector


load_dotenv()

app = FastAPI()

db_url = os.getenv("MONGODB_URI")
knowledge_base = TextKnowledgeBase(
    path="knowledge.txt",
    vector_db=MongoDBVector(
        collection_name="knowledge",
        db_url=db_url,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)

origins = [
    "http://localhost:5173",
]

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
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        tools=[DuckDuckGo()],
        structured_outputs=True,
        response_model=AnalyzeResponse,
    )
    return agent


def get_chat_agent(request_id: str):
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        session_id=request_id,
        show_tool_calls=True,
        search_knowledge=True,
        update_knowledge=True,
        knowledge_base=knowledge_base,
        add_chat_history_to_messages=True,
        read_chat_history=True,
        storage=MongoAgentStorage(
            collection_name="chat_history",
            db_url=db_url,
            db_name="product_analysis",
        ),
        instructions=[
            "Always search the knowledge base for the answer before using the tools.",
            "If you don't get the answer from the knowledge base, you can use the tools to search for the answer.",
        ],
        structured_outputs=True,
        response_model=ChatResponse,
    )


@app.post("/analyze-url")
async def analyze_url(request: AnalyzeURLRequest):
    response = analyze_image(request.url, request.request_id)
    return response


@app.post("/chat")
async def chat(request: ChatRequest):
    print("chat request_id", request.request_id)
    chat_agent = get_chat_agent(request.request_id)
    response = chat_agent.run(request.message)
    return response.content


def analyze_image(image_path, request_id):
    print("analyze_image request_id", request_id)
    agent = get_agent()
    response = agent.run(
        "Analyze the product image",
        images=[image_path],
    )

    text = prepare_knowledge_base(response)
    knowledge_base.load_text(text, upsert=True)

    return response.content


def prepare_knowledge_base(response: AnalyzeResponse):
    text = f"**Metadata**: \n\n"
    text += f"Product Name: {response.content.ProductName}\n"
    text += f"Ingredients: {', '.join(response.content.Ingredients)}\n"
    return text
