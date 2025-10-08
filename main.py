import streamlit as st
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------
# 🌿 TensorFlow Model Prediction Function
# ---------------------------------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# ---------------------------------------------------
# 🌾 Class Names (38 Classes)
# ---------------------------------------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# ---------------------------------------------------
# 🩺 Disease Information Dictionary
# ---------------------------------------------------
disease_info = {
    # (Same as your provided dictionary — unchanged)
    # Due to length, the dictionary is omitted here for brevity.
    # ✅ Make sure to keep your full `disease_info` as-is.
}


# ---------------------------------------------------
# 🏠 Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# ---------------------------------------------------
# 🏡 Home Page
# ---------------------------------------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)

    st.markdown("""
    Welcome to the **Plant Disease Recognition System!** 🌿🔍

    Upload an image of a plant, and our AI will detect possible diseases in seconds.

    ### How It Works
    1. **Upload Image:** Go to *Disease Recognition*.
    2. **Analyze:** Our model predicts the disease.
    3. **View Results:** Get prevention and treatment info instantly.

    ### Why Choose Us?
    - 🧠 **Accurate** — State-of-the-art ML model.
    - ⚡ **Fast** — Instant disease identification.
    - 💡 **User-Friendly** — Simple and intuitive.

    Click *Disease Recognition* to begin!
    """)


# ---------------------------------------------------
# ℹ️ About Page
# ---------------------------------------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### Dataset Information
    - Contains **87,000+ images** of healthy and diseased leaves.
    - 38 distinct classes.
    - Split: 80% training, 20% validation.
    - 33 test images for final evaluation.

    #### Structure
    1. `train/` → 70,295 images  
    2. `validation/` → 17,572 images  
    3. `test/` → 33 images  
    """)


# ---------------------------------------------------
# 🧠 Disease Recognition Page
# ---------------------------------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")

    # Upload Image
    test_image = st.file_uploader("Upload a leaf image:", type=["jpg", "jpeg", "png"])

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("🔍 **Analyzing...**")
            result_index = model_prediction(test_image)
            predicted_disease = CLASS_NAMES[result_index]
            st.success(f"🌾 Model Prediction: **{predicted_disease}**")

            # Display Treatment Info
            if predicted_disease in disease_info:
                info = disease_info[predicted_disease]
                st.subheader("🛡️ Prevention Techniques")
                st.write(info["prevention"])
                st.subheader("🌱 Organic Treatment")
                st.write(info["organic"])
                st.subheader("💊 Inorganic Treatment")
                st.write(info["inorganic"])
            else:
                st.info("No detailed prevention/treatment info available.")


    # ---------------------------------------------------
    # 📷 Camera Input Feature
    # ---------------------------------------------------
    st.markdown("---")
    st.subheader("📷 Capture Image from Camera")
    captured_image = st.camera_input("Take a picture of the leaf:")

    if captured_image:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        if st.button("Predict Captured Image"):
            st.write("🔍 **Analyzing captured image...**")
            result_index_cap = model_prediction(captured_image)
            predicted_cap = CLASS_NAMES[result_index_cap]
            st.success(f"🌾 Model Prediction (Captured): **{predicted_cap}**")


    # ---------------------------------------------------
    # 🧠 Feedback Collection & Logging
    # ---------------------------------------------------
    st.markdown("---")
    st.subheader("🧠 Feedback — Help Improve Model")

    feedback_label = st.text_input("Enter Correct Disease Name (e.g., Tomato___Late_blight)")

    if st.button("Submit Feedback"):
        source_image = None
        if captured_image is not None:
            source_image = captured_image
        elif test_image is not None:
            source_image = test_image

        if not feedback_label:
            st.warning("Please enter the correct label.")
        elif source_image is None:
            st.warning("No image selected or captured.")
        else:
            os.makedirs("feedback_data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{feedback_label}_{timestamp}.jpg"
            save_path = os.path.join("feedback_data", filename)

            img = Image.open(source_image)
            img.save(save_path)

            # Log to CSV
            log_path = "feedback_log.csv"
            new_row = pd.DataFrame([[timestamp, filename, feedback_label]],
                                   columns=["timestamp", "image_name", "correct_label"])
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                log_df = pd.concat([log_df, new_row], ignore_index=True)
            else:
                log_df = new_row
            log_df.to_csv(log_path, index=False)

            st.success(f"✅ Feedback saved: {filename}")


    # ---------------------------------------------------
    # 🔁 Retrain Model with Feedback Data
    # ---------------------------------------------------
    st.markdown("---")
    st.subheader("🔁 Retrain Model with Feedback Data")
    st.write("Fine-tune the model using corrected feedback samples and visualize training progress.")

    if st.button("Retrain Model"):
        st.info("⏳ Retraining started... please wait.")
        model = load_model("trained_model.keras")

        if os.path.exists("feedback_data") and len(os.listdir("feedback_data")) > 0:
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
            train_data = datagen.flow_from_directory(
                "feedback_data",
                target_size=(128, 128),
                batch_size=4,
                class_mode="categorical",
                subset="training"
            )
            val_data = datagen.flow_from_directory(
                "feedback_data",
                target_size=(128, 128),
                batch_size=4,
                class_mode="categorical",
                subset="validation"
            )

            model.compile(optimizer=Adam(learning_rate=1e-4),
                          loss="categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(train_data, validation_data=val_data, epochs=3, verbose=1)

            model.save("trained_model_updated.keras")
            st.success("✅ Model retrained successfully and saved as `trained_model_updated.keras`!")

            # 📊 Accuracy and Loss Visualization
            st.markdown("### 📈 Training Progress")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            ax[0].plot(history.history["accuracy"], label="Train Accuracy")
            ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
            ax[0].set_title("Accuracy Over Epochs")
            ax[0].set_xlabel("Epochs")
            ax[0].set_ylabel("Accuracy")
            ax[0].legend()

            ax[1].plot(history.history["loss"], label="Train Loss")
            ax[1].plot(history.history["val_loss"], label="Validation Loss")
            ax[1].set_title("Loss Over Epochs")
            ax[1].set_xlabel("Epochs")
            ax[1].set_ylabel("Loss")
            ax[1].legend()

            st.pyplot(fig)
        else:
            st.warning("⚠️ No feedback images found. Please submit corrections first.")
