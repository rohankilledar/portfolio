from fastapi import FastAPI, UploadFile
from typing import List, Dict, Any
import os
from openai import OpenAI
from ocr_helper import OCRHelper

app = FastAPI()
ocr_helper = OCRHelper()

openai_client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

@app.post("/upload")
async def upload_manual(file: UploadFile):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    text = ocr_helper.extract_text_from_pdf(file_path)
    print(f"Extracted text:\n{text[:1000]}...")  # Debugging: Print the first 1000 characters of the extracted text

    troubleshooting_info = await identify_troubleshooting_section(text)
    return troubleshooting_info

async def identify_troubleshooting_section(text: str) -> Dict[str, Any]:
    prompt = f"""The following text is extracted from a manual. Identify the troubleshooting section and extract information in JSON format. The JSON should include the page number on which the troubleshooting info is located, the problem, possible cause, and solution for each issue mentioned. If a page number isn't mentioned, infer it based on context.
    please make sure you keep the information from the below text and not make up new information. Only include troubleshooting information regarding the product.

{text}

Format:
[
  {{
    "page": <page_number>,
    "problem": "<problem_description>",
    "possible_cause": "<possible_cause>",
    "solution": "<solution>"
  }},
  ...
]
"""
    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )


    troubleshooting_section = response.choices[0].message.content
    print(f"ChatGPT API result: {troubleshooting_section}")  # Debugging: Print the API's result
    
    issues = parse_troubleshooting(troubleshooting_section)
    return {"troubleshooting": issues}

def parse_troubleshooting(section: str) -> List[Dict[str, Any]]:
    # Parse the JSON-like structure returned by ChatGPT
    try:
        issues = eval(section)  # Only use eval here if you trust the source. For untrusted sources, use json.loads
    except Exception as e:
        print(f"Error parsing troubleshooting section: {e}")
        issues = []

    return issues

@app.get("/")
def read_root():
    return {"message": "Welcome to the Troubleshooting Extractor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
