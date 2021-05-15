"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
import shutil
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        #img = Image.open(io.BytesIO(img_bytes))
        #results = model(img, size=640)
        #results.display(save=True, save_dir="static")
        results = get_prediction(img_bytes)
        #results.display(save=True)  # save as results1.jpg, results2.jpg... etc.
        results.display(save=True)
        #full_filename = os.path.join(app.config['RESULT_FOLDER'], 'image0.jpg')
        shutil.move("image0.jpg", "static/image0.jpg")
        #os.rename("image0.jpg", "/static/image0.jpg")
        return redirect("static/image0.jpg")
        #results.render()
    return render_template('index.html')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True).autoshape()  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
