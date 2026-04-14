import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

MODEL_PATH = Path("best_traffic_sign_model.keras")
LABELS_PATH = Path("labels.csv")
IMG_SIZE = (224, 224)
CLASS_NUMBERS = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3',
                 '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40',
                 '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51',
                 '52', '53', '54', '55', '56', '57', '6', '7', '8', '9']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    return pd.read_csv(LABELS_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("🚦 Traffic Sign Recognition")
st.write(
    "Upload a traffic sign image and the trained CNN model will predict the most likely class."
)

labels_df = load_labels()
model = load_model()

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            input_tensor = preprocess_image(image)
            predictions = model.predict(input_tensor, verbose=0)[0]
            top_indices = np.argsort(predictions)[::-1][:3]

        predicted_index = int(top_indices[0])
        predicted_class_id = int(CLASS_NUMBERS[predicted_index])
        predicted_label = labels_df.loc[labels_df["ClassId"] == predicted_class_id, "Name"].iloc[0]
        confidence = float(predictions[predicted_index])

        st.success(f"Prediction: {predicted_label}")
        st.metric("Confidence", f"{confidence:.2%}")

        st.subheader("Top 3 predictions")
        top_rows = []
        for idx in top_indices:
            class_id = int(CLASS_NUMBERS[idx])
            class_name = labels_df.loc[labels_df["ClassId"] == class_id, "Name"].iloc[0]
            top_rows.append({
                "Class ID": class_id,
                "Label": class_name,
                "Confidence": f"{float(predictions[idx]):.2%}"
            })

        st.table(pd.DataFrame(top_rows))

with st.expander("Model details"):
    st.write("Model file:", str(MODEL_PATH))
    st.write("Input image size:", f"{IMG_SIZE[0]} x {IMG_SIZE[1]}")
    st.write("Number of classes:", int(labels_df.shape[0]))
    st.write(
        "Note: the model already includes a Rescaling layer, so uploaded images are resized but not manually divided by 255 in the app."
    )


st.markdown("**Developed by IUIU CS Group 3:**")
st.markdown("""
- Farhan Segujja (224-0631-29800)
- Bugembe Mahad (224-0631-28843)
- Nanyunja Rahma (224-0631-29802)
""")