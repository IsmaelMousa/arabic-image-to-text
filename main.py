import io
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from utils import preprocess_image, model_loader, lookup_loader, predict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_path = "models/arabic_img2md_advance_ocr.h5"
chr_path = "models/char_to_num_layer.keras"
num_path = "models/num_to_char_layer.keras"


prediction_model = model_loader(model_path)
char_to_num, num_to_char = lookup_loader(chr_path, num_path)

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    new_image = preprocess_image(image)

    pred_text = predict(new_image, prediction_model, num_to_char, display=False)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": pred_text
    })
@app.post("api/predict/")
async def predict_image_api(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    new_image = preprocess_image(image)

    pred_text = predict(new_image, prediction_model, num_to_char)

    return {"prediction": pred_text}

