
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 載入訓練好的 CNN 模型
import os
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    filename="waste_model.keras"
)

model = load_model(model_path)

# 必須和訓練時的類別順序一致
class_names = ["PET", "carton", "glass"]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file and image_file.filename:
            try:
                # 讀取圖片並轉成 RGB
                image = Image.open(
                    io.BytesIO(image_file.read())
                ).convert("RGB")

                # 調整圖片尺寸
                image = image.resize((224, 224))

                # 轉成模型需要的陣列
                image_array = np.array(image, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)

                # 預測
                predictions = model.predict(image_array, verbose=0)
                predicted_index = int(np.argmax(predictions[0]))

                result = class_names[predicted_index]
                confidence = float(predictions[0][predicted_index]) * 100

            except Exception as e:
                result = f"圖片處理失敗：{e}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run()
