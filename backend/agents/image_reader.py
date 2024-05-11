from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "daekeun-ml/ko-trocr-base-nsmc-news-chatbot"
)
tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")

url = "https://raw.githubusercontent.com/aws-samples/sm-kornlp/main/trocr/sample_imgs/news_1.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
# img = Image.open(Path.cwd() / "service" / "apps" / "naver.jpg")
img.show()
pixel_values = processor(img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
