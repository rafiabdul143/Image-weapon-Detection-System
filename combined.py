import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title=" Weapon & Fire Detection System", layout="centered")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Constants
IMG_SIZE = 256




# Load models
weapon_model = tf.keras.models.load_model('caltech_normal.h5')
fire_model = load_model(r'C:\Users\Abdul Raqeeb\major_p\frontend rafi\firelp.keras', compile=False)

# Preprocessing functions
def preprocess_weapon_image(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_fire_image(image):
    image_np = np.array(image.convert("RGB"))  # Ensure PIL image is RGB
    image_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    image_array = np.expand_dims(image_resized / 255.0, axis=0)  # Normalize
    return image_array, image_resized

# Prediction functions

# Prediction functions
def predict_weapon(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    processed_img = cv2.resize(img, (224, 224))
    processed_img = processed_img.astype("float32") / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)

    # Predict
    box_output, class_output = weapon_model.predict(processed_img)
    class_index = np.argmax(class_output)
    confidence = np.max(class_output)

    class_labels = ["Weapon", "NoWeapon"]
    label = class_labels[class_index] if class_index < len(class_labels) else "Unknown"
    if confidence < 0.4:
        label = "Uncertain"

    # Bounding box
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_output[0]
    x1, y1 = int(w * x1), int(h * y1)
    x2, y2 = int(w * x2), int(h * y2)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if label == "weapon":
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({confidence:.2f})", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Prediction: {label}')
    plt.show()
    return label, confidence

    
def predict_fire(image):
    processed_image, display_image = preprocess_fire_image(image)
    prediction = fire_model.predict(processed_image)

    if prediction.shape[1] == 1:
        fire_prob = prediction[0][0]
        not_fire_prob = 1 - fire_prob
    else:
        not_fire_prob = prediction[0][0]
        fire_prob = prediction[0][1]

    threshold = 0.4
    if fire_prob > threshold:
        label = "🔥 Fire Detected"
        confidence = fire_prob
    elif not_fire_prob > threshold:
        label = "✅ Not Fire"
        confidence = not_fire_prob
    else:
        label = "❓ Uncertain"
        confidence = max(fire_prob, not_fire_prob)

    return label, confidence, display_image, fire_prob, not_fire_prob

# Streamlit UI
st.markdown('<h1 class="st_title"> Weapon & Fire Detection System</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
   .custom-alert {
        background-color: rgba(128, 128, 128, 0.172);
        color: white;
        padding: 12px 20px;
        border: 2px solid red;  /* Red border added */
        border-radius: 0 0 12px 12px;
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
    }
    </style>
    <div class="custom-alert">
        📸 Upload a clear image. Models work best with centered and well-lit subjects.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="marquee-container">
        <div class="marquee-text">
            📢🚨 Fire Detection: Custom CNN architecture built from scratch using Keras – optimized for identifying fire in real-time scenes.| 🛡️ Weapon Detection: Leveraged NASNet as a feature extractor in a Multi-Output Keras model. Integrated Bounding Box Regression to classify and localize weapons (e.g., knives, guns) with high precision.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



option = st.radio("Choose Detection Type", ("Weapon Detection", "Fire Detection"))


uploaded_file = st.file_uploader("📤 Hello, please upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

if option == "Weapon Detection":
    # Weapon detection functionality
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    if st.button('⚠️Detect Weapon'):
        label, confidence = predict_weapon(image)

        if label == "Uncertain":
            st.warning(f"⚠️ Low confidence: {confidence:.2f}. Unable to determine if a weapon is present.")
        else:
            st.success(f"Prediction: {label} with confidence: {confidence:.2f}")


   
elif option == "Fire Detection":
        if st.button('🔥 Detect Fire'):
            label, confidence, display_img, fire_prob, not_fire_prob = predict_fire(image)
            st.subheader(f"🧠 Model Prediction: **{label}**")
            st.image(display_img, caption=f"🔥 Fire Probability: {fire_prob:.2%} | ✅ Not Fire: {not_fire_prob:.2%}", use_container_width=True)
