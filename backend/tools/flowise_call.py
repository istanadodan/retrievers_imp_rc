import requests

API_URL = "http://localhost:3000/api/v1/prediction/215b0705-692f-4fa2-a460-adbe967a7556"


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()


output = query(
    {
        "question": "what is 3 plus 2? use tool.",
    }
)

print(output)
