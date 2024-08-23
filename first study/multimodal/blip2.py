from PIL import Image

import requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(

    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16

)  # doctest: +IGNORE_RESULT

url = "https://as1.ftcdn.net/v2/jpg/00/05/26/38/1000_F_5263801_wbQt53pDJu0XHbIbplJYoPRcq0z30pin.jpg"

image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(generated_text)