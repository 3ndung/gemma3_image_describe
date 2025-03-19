from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
import ollama
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_gemma_icon():
    with open("./static/gemma3.png", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "ocr_result_html": None,
        "image_base64": None,
        "error": None,
        "gemma_icon": get_gemma_icon()
    })

@app.post("/upload")
async def describe_image(request: Request, file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode()

        # Analyze image with Gemma-3 Vision
        response = ollama.chat(
            model='gemma3:latest',
            messages=[{
                'role': 'user',
                'content': """Analyze and describe this image in detail. Include:
                            - Main subjects and objects
                            - Colors and visual style
                            - Any visible text
                            - Overall scene context
                            - Emotional tone or atmosphere
                            Provide your response in clear, natural English.""",
                'images': [image_data]
            }]
        )
        description_result = response.message.content
        error = None
    except Exception as e:
        description_result = None
        error = f"Error processing image: {str(e)}"
        image_base64 = None
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "ocr_result_html": description_result,  # Keeping original variable name for template compatibility
        "image_base64": image_base64,
        "error": error,
        "gemma_icon": get_gemma_icon()
    })