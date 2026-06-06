import os
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
from enum import Enum
from openai import AsyncOpenAI, OpenAI

load_dotenv()


class Config:

    github_token = os.getenv("GITHUB_TOKEN")
    openai_key=os.getenv("OPENAI_API_KEY")

    MODEL = "gpt-4o-mini"

    IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID")
    ACCESS_TOKEN = os.environ.get("IG_ACCESS_TOKEN")

    github_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=github_token)

    github_model = OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=github_client
    )
     
    image_model = "black-forest-labs/flux-2-pro"