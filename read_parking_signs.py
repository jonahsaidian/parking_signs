# %%
from roboflow import Roboflow
from roboflow.models.object_detection import ObjectDetectionModel
from PIL import Image
import cv2 as cv
import numpy as np

try:
    from api_key import (
        API_KEY,
    )  # file is not committed for security reasons please add your own key
except ModuleNotFoundError:
    API_KEY = ""
import easyocr
import argparse

reader = easyocr.Reader(["en"])
rf = Roboflow(api_key=API_KEY)
# %%
parser = argparse.ArgumentParser(
    description="Command line interface for parking sign detection and processing"
)
parser.add_argument(
    "image filepath", type=str, help="local file path to the image to be analyzed"
)

parser.add_argument(
    "API_KEY",
    type=str,
    help="optional API key overwrite, if you have not set up your API Key in the filepath use this argument to provide it instead",
)
# %%
project = rf.workspace().project("read_parking_signs")
model = project.version("1").model
# %%
# test the model on a local image, if set to None skip
# test_file = "./no_parking.jpg"
test_file = None
if test_file:
    print(model.predict(test_file, confidence=40, overlap=30).json())
    # visualize your prediction
    model.predict(test_file, confidence=40, overlap=30).save("prediction.jpg")
    pred = model.predict(test_file, confidence=40, overlap=30).json()["predictions"][0]
# %%
def detect_and_trim_image(fp: str, model: ObjectDetectionModel):
    """
    Detect signs in a local image and isolate them
    """
    preds = model.predict(fp, confidence=40, overlap=30).json()["predictions"]
    img = Image.open(fp)
    results = []
    for pred in preds:
        if pred["confidence"] < 0.70 or not pred["class"] == "signs":
            continue
        box = (
            pred["x"] - pred["width"] / 2,
            pred["y"] - pred["height"] / 2,
            pred["x"] + pred["width"] / 2,
            pred["y"] + pred["height"] / 2,
        )
        img2 = img.crop(box)
        results.append(img2)
    img.close()
    return results


def read_signs(signs_list: list[Image.Image], reader: easyocr.Reader):
    """
    for a provided list of sign images read them using the OCR reader and return a list of the text in each sign
    """
    if not signs_list:
        return
    sign_text = []
    for sign in signs_list:
        cv_sign = cv.cvtColor(np.array(sign), cv.COLOR_RGB2BGR)
        text = reader.readtext(cv_sign)
        sign_text.append(text)
    full_text = []
    for text in sign_text:
        full_text.append("")
        for words in text:
            full_text[-1] += words[1] + " "
        else:
            full_text[-1] = full_text[-1][:-1]
    return full_text


# %%
# test the functions if test_file is provided
if test_file:
    signs_list = detect_and_trim_image(test_file)
    if signs_list:
        signs_list[0].show()
    full_text = read_signs(signs_list)
    print(full_text)

# %%
# TODO: feed this text to a LLM or other ML application to determine if parking is available

# %%
def __main__():
    # import arguments
    args = parser.parse_args()
    filepath = args[0]
    if args[1]:
        API_KEY = args[1]
    if not API_KEY:
        raise PermissionError("No valid API key was provided")

    # import RoboFlow model
    project = rf.workspace().project("read_parking_signs")
    model = project.version("1").model
    signs_list = detect_and_trim_image(filepath, model)

    full_text = read_signs(signs_list, reader)
    print("The text from the provided signs is:")
    for text in full_text:
        print(text)
        print("\n")


# %%
__main__()
# %%
