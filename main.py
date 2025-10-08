import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from datetime import datetime

# -----------------------------
# Tensorflow Model Prediction
# -----------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -----------------------------
# Retrain Model Function
# -----------------------------
def retrain_model(feedback_dir="feedback_data"):
    MODEL_PATH = "trained_model.keras"
    UPDATED_MODEL_PATH = "trained_model_updated.keras"

    if not os.path.exists(feedback_dir):
        st.warning("⚠️ No feedback data found!")
        return

    images, labels = [], []
    for file in os.listdir(feedback_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            label = file.split("_")[0]
            img_path = os.path.join(feedback_dir, file)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)

    if len(images) == 0:
        st.warning("⚠️ No feedback images found for retraining.")
        return

    st.info(f"🧠 Loaded {len(images)} feedback images for retraining...")

    X = np.array(images)
    y_labels = np.array(labels)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)
    y_categorical = to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    st.info("🚀 Retraining model with new feedback data... This may take a few minutes ⏳")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        validation_data=(X_val, y_val),
        epochs=5,
        verbose=1
    )

    model.save(UPDATED_MODEL_PATH)

    log_file = os.path.join(feedback_dir, "retrain_log.txt")
    with open(log_file, "a") as log:
        log.write(f"Retrained on {len(images)} samples | {datetime.now()}\n")

    st.success("✅ Model retrained successfully and saved as 'trained_model_updated.keras'")
    st.write("📊 Training Accuracy:", round(history.history['accuracy'][-1] * 100, 2), "%")
    st.write("📉 Validation Accuracy:", round(history.history['val_accuracy'][-1] * 100, 2), "%")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Feedback History", "Model Retraining"])

# -----------------------------
# Home Page
# -----------------------------
if app_mode == "Home":
    st.header("🌾 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System** 🌱

    ### ✨ Features
    - 📸 Capture crop image directly using camera
    - 🧠 AI-powered disease detection
    - 💬 Feedback system that helps model improve
    - 🔄 Retrain model with your feedback data

    Go to the **Disease Recognition** page to get started!
    """)

# -----------------------------
# About Page
# -----------------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### 📚 Dataset Overview
    - 87,000+ RGB images of healthy and diseased leaves  
    - 38 crop classes  
    - Train/Validation split: 80/20  
    - Based on the **PlantVillage Dataset**

    #### 👨‍💻 Team Goals
    - Develop an intelligent, scalable disease detection tool  
    - Help farmers detect crop diseases early  
    - Continuously improve model accuracy through user feedback  
    """)

# -----------------------------
# Disease Recognition Page
# -----------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Plant Disease Detection")

    st.markdown("### Choose Input Source:")
    source = st.radio("Select input method:", ["📁 Upload Image", "📸 Capture from Camera"])

    test_image = None
    if source == "📁 Upload Image":
        test_image = st.file_uploader("Upload a leaf image:", type=["jpg", "jpeg", "png"])
    elif source == "📸 Capture from Camera":
        test_image = st.camera_input("Take a picture of your crop leaf:")

    if test_image is not None:
        st.image(test_image, use_column_width=True, caption="Selected Image")

        if st.button("🔍 Predict Disease"):
            st.write("Analyzing image... please wait ⏳")
            result_index = model_prediction(test_image)

            # All 38 Classes
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            predicted_disease = class_name[result_index]
            st.success(f"🌿 Model Prediction: **{predicted_disease}**")

            # --------------- FEEDBACK SYSTEM ---------------
            st.subheader("🧠 Model Feedback")
            correct_label = st.selectbox("If prediction is wrong, select correct disease:", class_name)
            if st.button("✅ Submit Feedback"):
                feedback_dir = "feedback_data"
                os.makedirs(feedback_dir, exist_ok=True)
                feedback_file = os.path.join(feedback_dir, "feedback_log.csv")

                with open(feedback_file, "a") as f:
                    f.write(f"{datetime.now()},{predicted_disease},{correct_label}\n")

                # Save image for retraining
                image_save_path = os.path.join(feedback_dir, f"{correct_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                with open(image_save_path, "wb") as img_file:
                    img_file.write(test_image.getbuffer())

                st.success("✅ Feedback saved successfully! Thank you for helping improve the model.")

# -----------------------------
# Feedback History Page
# -----------------------------
elif app_mode == "Feedback History":
    st.header("📋 Feedback History")
    feedback_file = os.path.join("feedback_data", "feedback_log.csv")
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            data = f.readlines()
            st.write("### Recent Feedback Entries:")
            for line in data[-20:]:
                timestamp, predicted, correct = line.strip().split(",")
                st.write(f"🕒 {timestamp} | ❌ Predicted: {predicted} | ✅ Correct: {correct}")
    else:
        st.info("No feedback submitted yet!")

# -----------------------------
# Model Retraining Page
# -----------------------------
elif app_mode == "Model Retraining":
    st.header("🔄 Model Retraining Dashboard")
    st.write("Use feedback data to improve model accuracy over time.")

    feedback_dir = "feedback_data"
    feedback_file = os.path.join(feedback_dir, "feedback_log.csv")

    if os.path.exists(feedback_file):
        num_feedback = len([f for f in os.listdir(feedback_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        st.write(f"🧾 Feedback Samples Available: **{num_feedback}**")
    else:
        st.info("No feedback available yet. Please collect data first.")

    if st.button("🚀 Retrain Model Now"):
        retrain_model(feedback_dir)
