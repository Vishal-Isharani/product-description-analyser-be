from tempfile import NamedTemporaryFile
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from constants import SYSTEM_PROMPT, INSTRUCTIONS
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

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


class AnalyzeResponse(BaseModel):
    Ingredients: list[str]
    HealthImplications: list[str]
    Considerations: list[str]
    NutritionInformation: list[str]
    Conclusion: str


def get_agent():
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Food Product Analyzer",
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        tools=[DuckDuckGo()],
        structured_outputs=True,
        response_model=AnalyzeResponse,
        # markdown=True,
    )


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/analyze")
async def analyze(file: UploadFile):
    temp_path = await save_to_temp_file(file)
    response = analyze_image(temp_path)
    os.unlink(temp_path)
    return response


@app.post("/analyze-url")
async def analyze_url(request: AnalyzeURLRequest):
    print(request.url)
    response = analyze_image(request.url)
    return response


async def save_to_temp_file(file):
    suffix = f".{file.content_type.split('/')[-1]}"
    with NamedTemporaryFile(dir=".", suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_file_path = tmp.name  # Save the temp file path
        return temp_file_path


def analyze_image(image_path):
    agent = get_agent()
    response = agent.run(
        "Analyze the product image",
        images=[image_path],
    )
    return response.content
