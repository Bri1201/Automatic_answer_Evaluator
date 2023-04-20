from google.cloud import vision
import io
import os

def detect_text(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    description = texts[0].description
    return description
