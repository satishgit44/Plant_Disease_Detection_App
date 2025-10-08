import streamlit as st
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# -----------------------------
# Home Page
# -----------------------------
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

# -----------------------------
# About Page
# -----------------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
     #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
    #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

    """)

# -----------------------------
# Disease Recognition Page
# -----------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")

    # 🔹 Added Feature 1: Option to Capture Image from Camera or Upload
    st.markdown("### Choose Image Input Source:")
    source_option = st.radio("Select input method:", ["📁 Upload from Device", "📸 Capture from Camera"])

    test_image = None
    if source_option == "📁 Upload from Device":
        test_image = st.file_uploader("Upload a leaf image:", type=["jpg", "jpeg", "png"])
    elif source_option == "📸 Capture from Camera":
        test_image = st.camera_input("Capture Image")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("🔍 **Analyzing...**")
            result_index = model_prediction(test_image)

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
            st.success(f"🌾 Model Prediction: **{predicted_disease}**")

            # -----------------------------
            # 🔹 Added Feature 2: Feedback System for Model Learning
            # -----------------------------
            st.subheader("🧠 Feedback (Model Learning System)")
            st.write("If the above prediction seems incorrect, please select the correct disease label below:")

            correct_label = st.selectbox("Select Correct Disease Label:", class_name)
            if st.button("✅ Submit Feedback"):
                feedback_dir = "feedback_data"
                os.makedirs(feedback_dir, exist_ok=True)

                # Save feedback log
                feedback_file = os.path.join(feedback_dir, "feedback_log.csv")
                with open(feedback_file, "a") as f:
                    f.write(f"{datetime.now()},{predicted_disease},{correct_label}\n")

                # Save image with correct label for future retraining
                image_save_path = os.path.join(feedback_dir, f"{correct_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                with open(image_save_path, "wb") as img_file:
                    img_file.write(test_image.getbuffer())

                st.success("✅ Feedback submitted successfully! This will help improve model accuracy over time.")

            st.info("""
            ⚙️ *Note:* Your feedback image and label are stored locally in the 'feedback_data' folder.
            These will be used later to retrain and fine-tune the model periodically.
            """)

            # -----------------------------
            # Disease Info Dictionary (All 38 Classes)
            # -----------------------------
            disease_info = {

                # Apple
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

                # Blueberry
                'Blueberry___healthy': {
                    "prevention": "Maintain soil pH (4.5–5.5), prune regularly, and irrigate properly.",
                    "organic": "Compost mulch and neem oil spray prevent fungal issues.",
                    "inorganic": "No chemical treatment needed."
                },

                # Add rest of your existing 38 disease info dictionary entries...
            }

            # -----------------------------
            # Show Treatment Info
            # -----------------------------
            if predicted_disease in disease_info:
                info = disease_info[predicted_disease]
                st.subheader("🛡️ Prevention Techniques:")
                st.write(info["prevention"])
                st.subheader("🌱 Organic Treatment:")
                st.write(info["organic"])
                st.subheader("💊 Inorganic Treatment:")
                st.write(info["inorganic"])
            else:
                st.info("No detailed prevention/treatment information available for this plant.")
