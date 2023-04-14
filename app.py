import os
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradeAnswer import grade_answer

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/model_executor', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        keywords = request.form.get("keywords").split(",")
        model_answer = request.form.get("model_answer")
        student_answer = request.form.get("student_answer")

        if "student_answer_image" in request.files:
            file = request.files["student_answer_image"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                student_answer = pytesseract.image_to_string(Image.open(filepath))

        score = grade_answer(keywords, model_answer, student_answer)

        return jsonify({"score": score})

    return render_template("NLPModel.html")

if __name__ == "__main__":
    app.run(debug=True)