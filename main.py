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
# 🩺 Disease Information Dictionary (Full)
# ---------------------------------------------------
disease_info = {
    'Apple___Apple_scab': {
        "prevention": "Remove fallen leaves, prune infected branches, and apply fungicides.",
        "organic": "Use neem oil or sulfur sprays weekly.",
        "inorganic": "Apply mancozeb or captan-based fungicide."
    },
    'Apple___Black_rot': {
        "prevention": "Prune out dead wood, remove mummified fruit, and use resistant varieties.",
        "organic": "Use copper-based sprays every 10 days.",
        "inorganic": "Use thiophanate-methyl or mancozeb fungicides."
    },
    'Apple___Cedar_apple_rust': {
        "prevention": "Avoid planting near juniper trees; remove galls from cedar trees.",
        "organic": "Apply sulfur or copper fungicide before infection period.",
        "inorganic": "Use myclobutanil or propiconazole spray."
    },
    'Apple___healthy': {
        "prevention": "Maintain good orchard hygiene and monitor regularly.",
        "organic": "Apply neem oil occasionally as preventive.",
        "inorganic": "No treatment needed."
    },
    # (💡 truncated here for brevity in this message — keep all your original disease_info content)
    # You can safely copy-paste your entire original disease_info dictionary here exactly as it was.
}


# ---------------------------------------------------
# 🏠 Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# ---------------------------------------------------
# HOME PAGE (original markdown restored)
# ---------------------------------------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
   Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)


# ---------------------------------------------------
# ABOUT PAGE (full markdown restored)
# ---------------------------------------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
     #### About Dataset
     This dataset is recreated using offline augmentation from the original dataset.
     The original dataset can be found on this GitHub repo.
     This dataset consists of about 87K RGB images of healthy and diseased crop leaves
     categorized into 38 different classes.
     
     The dataset is divided into an 80/20 ratio of training and validation sets,
     preserving the directory structure. A new directory containing 33 test images is created later for prediction.

     #### Content
     1. **train** (70,295 images)  
     2. **test** (33 images)  
     3. **validation** (17,572 images)
    """)


# ---------------------------------------------------
# DISEASE RECOGNITION PAGE
# ---------------------------------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")

    # ---------- New Feature: User Chooses Input Method ----------
    st.subheader("📸 Choose Image Input Method")
    image_option = st.radio("Select Option:", ("📁 Upload Image", "📷 Capture Image"))

    test_image = None
    captured_image = None

    if image_option == "📁 Upload Image":
        test_image = st.file_uploader("Upload a leaf image:", type=["jpg", "jpeg", "png"])
        if test_image is not None:
            st.image(test_image, use_column_width=True, caption="Uploaded Image")
            if st.button("Predict Uploaded Image"):
                st.write("🔍 **Analyzing...**")
                result_index = model_prediction(test_image)
                predicted_disease = CLASS_NAMES[result_index]
                st.success(f"🌾 Model Prediction: **{predicted_disease}**")

                # Show Treatment Info
                if predicted_disease in disease_info:
                    info = disease_info[predicted_disease]
                    st.subheader("🛡️ Prevention Techniques:")
                    st.write(info["prevention"])
                    st.subheader("🌱 Organic Treatment:")
                    st.write(info["organic"])
                    st.subheader("💊 Inorganic Treatment:")
                    st.write(info["inorganic"])
                else:
                    st.info("No detailed prevention/treatment information available.")

    elif image_option == "📷 Capture Image":
        captured_image = st.camera_input("Take a picture of the leaf:")
        if captured_image is not None:
            st.image(captured_image, caption="Captured Image", use_column_width=True)
            if st.button("Predict Captured Image"):
                st.write("🔍 **Analyzing captured image...**")
                result_index_cap = model_prediction(captured_image)
                predicted_cap = CLASS_NAMES[result_index_cap]
                st.success(f"🌾 Model Prediction (Captured): **{predicted_cap}**")

                if predicted_cap in disease_info:
                    info = disease_info[predicted_cap]
                    st.subheader("🛡️ Prevention Techniques:")
                    st.write(info["prevention"])
                    st.subheader("🌱 Organic Treatment:")
                    st.write(info["organic"])
                    st.subheader("💊 Inorganic Treatment:")
                    st.write(info["inorganic"])

    # ----------------------------- Feedback System + CSV Logging -----------------------------
    st.markdown("---")
    st.subheader("🧠 Feedback — Help Improve Model")
    feedback_label = st.text_input("Enter Correct Disease Name (e.g., Tomato___Late_blight)")

    if st.button("Submit Feedback"):
        source_image = captured_image if captured_image else test_image
        if not feedback_label:
            st.warning("Please enter the correct label first.")
        elif source_image is None:
            st.warning("No image selected or captured.")
        else:
            os.makedirs("feedback_data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{feedback_label}_{timestamp}.jpg"
            save_path = os.path.join("feedback_data", filename)

            img = Image.open(source_image)
            img.save(save_path)

            # Log feedback in CSV
            log_path = "feedback_log.csv"
            new_row = pd.DataFrame([[timestamp, filename, feedback_label]],
                                   columns=["timestamp", "image_name", "correct_label"])
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                log_df = pd.concat([log_df, new_row], ignore_index=True)
            else:
                log_df = new_row
            log_df.to_csv(log_path, index=False)
            st.success(f"✅ Feedback saved and logged: {filename}")

    # ----------------------------- Developer-only Retrain Section -----------------------------
    st.markdown("---")
    st.subheader("🔐 Developer Access Only")
    dev_key = st.text_input("Enter Developer Key to Access Retraining:", type="password")

    if dev_key == "admin123":  # change this to your private password
        st.success("✅ Developer access granted.")
        st.subheader("🔁 Retrain Model with Feedback Data")
        st.write("Fine-tune the model using newly corrected samples and visualize training progress.")

        if st.button("Retrain Model"):
            st.info("⏳ Retraining started... please wait a moment.")
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

                model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(train_data, validation_data=val_data, epochs=3, verbose=1)

                model.save("trained_model_updated.keras")
                st.success("✅ Model retrained successfully and saved as `trained_model_updated.keras`!")

                # 📊 Visualize Accuracy and Loss
                st.markdown("### 📈 Training Progress")
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))

                ax[0].plot(history.history["accuracy"], label="Train Accuracy")
                ax[0].plot(history.history["val_accuracy"], label="Val Accuracy")
                ax[0].set_title("Accuracy Over Epochs")
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Accuracy")
                ax[0].legend()

                ax[1].plot(history.history["loss"], label="Train Loss")
                ax[1].plot(history.history["val_loss"], label="Val Loss")
                ax[1].set_title("Loss Over Epochs")
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Loss")
                ax[1].legend()

                st.pyplot(fig)
            else:
                st.warning("⚠️ No feedback images found. Please add corrections before retraining.")
    elif dev_key:
        st.error("❌ Invalid Developer Key.")
