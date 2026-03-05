import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Load Model (only once)
# ----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("malaria_detector.h5")

model = load_cnn_model()

# Image shape from your training
IMAGE_SIZE = (130, 130)

st.title("🦠 Malaria Cell Detection")
st.write("Upload a cell image to check if it is Parasitized or Uninfected.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result:")

    if prediction < 0.5:
        st.error("🛑 Parasitized (Malaria Infected)")
    else:
        st.success("✅ Uninfected Cell")

    st.write(f"Raw Prediction Score: {prediction:.4f}")