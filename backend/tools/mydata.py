import datetime
import json
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool


@tool
def get_mydata_company_info(corp_code: str):
    """질의문에 나오는 회사코드를 받아, 해당 회사관련정보를 조회하여 반환한다"""
    company_info = {
        "corp_code": corp_code,
    }
    # Example output returned from an API or database
    if corp_code == "12345":
        company_info = {
            "corp_code": corp_code,
            "corp_name": "Example Corp",
            "corp_type": "Public Limited Company",
            "establish_date": "2020-01-01",
            "address": "123 Main St, Anytown, USA",
            "phone": "555-1234",
            "email": "example@example.com",
        }

    return json.dumps(company_info)
