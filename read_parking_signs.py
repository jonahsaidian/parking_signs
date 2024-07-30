# %%
from roboflow import Roboflow
from PIL import Image
import pytesseract
from api_key import API_KEY # file is not committed for security reasons please add your own key
# %%
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("read_parking_signs")
model = project.version("1").model
test_file = "./no_parking.jpg"
# %%
# test the model on a local image
print(model.predict(test_file, confidence=40, overlap=30).json())
# visualize your prediction
model.predict(test_file, confidence=40, overlap=30).save("prediction.jpg")
pred = model.predict(test_file, confidence=40, overlap=30).json()["predictions"][0]
# %%

def predict_and_trim_image(fp:str):
    """
    Detect signs in a local image and isolate them
    """
    preds = model.predict(test_file, confidence=40, overlap=30).json()["predictions"]
    img = Image.open(test_file)
    results = []
    for pred in preds:
        if pred["confidence"]<0.70 or not pred["class"]=="signs":
            continue
        box = (pred["x"] - pred["width"]/2, pred["y"] - pred["height"]/2, pred["x"] + pred["width"]/2, pred["y"] + pred["height"]/2)
        img2 = img.crop(box)
        results.append(img2)
    img.close()
    return results

# %%
# test the function
signs_list = predict_and_trim_image(test_file)
if signs_list:
    signs_list[0].show()

# %%
# use ocr to read the signs
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract

sign_text = []
for sign in signs_list:
    text = pytesseract.image_to_string(sign,lang="eng") 
    sign_text.append(text)
print(sign_text)
# %%
