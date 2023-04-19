import keras_ocr

def ocr_handwritten_text_keras(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    image = keras_ocr.tools.read(image_path)
    predictions = pipeline.recognize([image])[0]
    text = " ".join([text for _, text in predictions])
    return text
