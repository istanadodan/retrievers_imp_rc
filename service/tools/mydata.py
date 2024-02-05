import datetime
import json
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool


@tool
def get_mydata_company_info(corp_code: str):
    """Get company information of the corporate code"""

    # Example output returned from an API or database
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
