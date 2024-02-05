from typing import Dict, List
from serpapi import GoogleSearch
from langchain_community.tools.google_search import GoogleSearchRun
from dotenv import load_dotenv
import os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
import json

load_dotenv()


@tool("serpapi")
def get_query_from_serpapi(query: str) -> List[Dict[str, object]]:
    """gather news articles about user's query from webs using serpapi"""
    params = {
        "q": query,
        "location": "Seoul, Seoul, South Korea",
        "hl": "ko",
        "gl": "kr",
        "google_domain": "google.co.kr",
        "api_key": os.getenv("SERP_API_KEY"),
        "tbm": "nws",
        "num": 3,
    }

    search = GoogleSearch(params)
    result = search.get_dict()
    # print(results)
    news = list(
        map(
            lambda x: x.pop("thumbnail"),
            result.get("news_results", []),
        )
    )
    return json.dumps(news)


if __name__ == "__main__":
    get_query_from_serpapi("high performance i7 computer")
    print("done")
