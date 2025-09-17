from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

@app.route('/webhook', methods=['POST', 'GET'])
async def webhook(request:Request):
    logging.info(f"Received webhook: {request=}")
    data = await request.body()
    logging.info("Received webhook data: %s", data)
    return JSONResponse(content={"status": "success"})

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("webhook:app", host="0.0.0.0", port=3000, reload=True)
