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

                # Cherry
                'Cherry_(including_sour)___Powdery_mildew': {
                    "prevention": "Ensure airflow by pruning and avoid overhead watering.",
                    "organic": "Use sulfur-based sprays or neem oil.",
                    "inorganic": "Apply myclobutanil or trifloxystrobin fungicide."
                },
                'Cherry_(including_sour)___healthy': {
                    "prevention": "Avoid excessive nitrogen and maintain air circulation.",
                    "organic": "Periodic neem oil sprays.",
                    "inorganic": "No treatment needed."
                },

                # Corn
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
                    "prevention": "Rotate crops, use resistant varieties, and destroy infected residue.",
                    "organic": "Compost teas or neem oil applications.",
                    "inorganic": "Use azoxystrobin or pyraclostrobin fungicides."
                },
                'Corn_(maize)___Common_rust_': {
                    "prevention": "Use rust-resistant hybrids and crop rotation.",
                    "organic": "Neem oil spray every 7 days.",
                    "inorganic": "Spray mancozeb or propiconazole fungicides."
                },
                'Corn_(maize)___Northern_Leaf_Blight': {
                    "prevention": "Use resistant hybrids and ensure balanced fertilization.",
                    "organic": "Garlic extract or neem oil foliar spray.",
                    "inorganic": "Apply fungicides with azoxystrobin or mancozeb."
                },
                'Corn_(maize)___healthy': {
                    "prevention": "Maintain spacing and nutrient management.",
                    "organic": "Use compost tea as foliar feed.",
                    "inorganic": "No chemical needed."
                },

                # Grape
                'Grape___Black_rot': {
                    "prevention": "Remove infected leaves and prune vines.",
                    "organic": "Sulfur dust or neem oil spray.",
                    "inorganic": "Spray mancozeb or myclobutanil fungicide."
                },
                'Grape___Esca_(Black_Measles)': {
                    "prevention": "Avoid wounds on vines and disinfect pruning tools.",
                    "organic": "Use Trichoderma bio-control fungus.",
                    "inorganic": "Apply systemic fungicide like tebuconazole."
                },
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
                    "prevention": "Remove infected leaves and maintain vineyard hygiene.",
                    "organic": "Neem oil or baking soda spray.",
                    "inorganic": "Use chlorothalonil fungicide."
                },
                'Grape___healthy': {
                    "prevention": "Ensure good air flow and regular pruning.",
                    "organic": "Periodic neem oil spray.",
                    "inorganic": "No treatment needed."
                },

                # Orange
                'Orange___Haunglongbing_(Citrus_greening)': {
                    "prevention": "Control psyllid vector and remove infected trees.",
                    "organic": "Neem oil for psyllid control.",
                    "inorganic": "Imidacloprid spray to control psyllids."
                },

                # Peach
                'Peach___Bacterial_spot': {
                    "prevention": "Use disease-free nursery plants and avoid overhead watering.",
                    "organic": "Copper oxychloride spray weekly.",
                    "inorganic": "Apply streptomycin-based bactericides."
                },
                'Peach___healthy': {
                    "prevention": "Regular pruning and balanced fertilization.",
                    "organic": "Neem oil preventive spray.",
                    "inorganic": "No chemical needed."
                },

                # Pepper
                'Pepper,_bell___Bacterial_spot': {
                    "prevention": "Use resistant seeds and copper sprays during wet periods.",
                    "organic": "Copper-based fungicide every 10 days.",
                    "inorganic": "Apply streptomycin spray."
                },
                'Pepper,_bell___healthy': {
                    "prevention": "Avoid leaf wetness and remove debris.",
                    "organic": "Neem oil or compost tea.",
                    "inorganic": "No treatment needed."
                },

                # Potato
                'Potato___Early_blight': {
                    "prevention": "Avoid overhead watering, and rotate crops yearly.",
                    "organic": "Neem oil or compost tea spray.",
                    "inorganic": "Use chlorothalonil or mancozeb fungicide."
                },
                'Potato___Late_blight': {
                    "prevention": "Avoid moisture and plant spacing.",
                    "organic": "Use copper oxychloride spray.",
                    "inorganic": "Metalaxyl or cymoxanil fungicides."
                },
                'Potato___healthy': {
                    "prevention": "Healthy soil and crop rotation.",
                    "organic": "Trichoderma-based soil application.",
                    "inorganic": "No chemical required."
                },

                # Raspberry
                'Raspberry___healthy': {
                    "prevention": "Ensure airflow and remove old canes.",
                    "organic": "Neem or sulfur spray occasionally.",
                    "inorganic": "No treatment needed."
                },

                # Soybean
                'Soybean___healthy': {
                    "prevention": "Rotate crops and avoid excessive irrigation.",
                    "organic": "Compost tea and neem-based tonic.",
                    "inorganic": "No treatment needed."
                },

                # Squash
                'Squash___Powdery_mildew': {
                    "prevention": "Improve air flow and avoid dense planting.",
                    "organic": "Spray milk solution (1:10) or sulfur dust.",
                    "inorganic": "Use myclobutanil or trifloxystrobin fungicides."
                },

                # Strawberry
                'Strawberry___Leaf_scorch': {
                    "prevention": "Remove infected leaves and improve drainage.",
                    "organic": "Neem oil spray weekly.",
                    "inorganic": "Use captan-based fungicide."
                },
                'Strawberry___healthy': {
                    "prevention": "Avoid excess moisture and mulch properly.",
                    "organic": "Compost extract foliar spray.",
                    "inorganic": "No treatment needed."
                },

                # Tomato
                'Tomato___Bacterial_spot': {
                    "prevention": "Use disease-free seeds and copper-based sprays.",
                    "organic": "Neem or copper fungicide weekly.",
                    "inorganic": "Use streptomycin or copper hydroxide."
                },
                'Tomato___Early_blight': {
                    "prevention": "Remove infected leaves and avoid wetting foliage.",
                    "organic": "Apply compost tea or copper fungicide.",
                    "inorganic": "Use chlorothalonil or mancozeb-based fungicides."
                },
                'Tomato___Late_blight': {
                    "prevention": "Avoid excess humidity and waterlogging.",
                    "organic": "Use neem oil or copper spray.",
                    "inorganic": "Apply metalaxyl or cymoxanil-based fungicide."
                },
                'Tomato___Leaf_Mold': {
                    "prevention": "Improve air circulation and reduce humidity.",
                    "organic": "Spray baking soda (1 tsp/L water).",
                    "inorganic": "Use chlorothalonil fungicide."
                },
                'Tomato___Septoria_leaf_spot': {
                    "prevention": "Remove infected debris and mulch base area.",
                    "organic": "Copper-based spray weekly.",
                    "inorganic": "Use mancozeb or chlorothalonil."
                },
                'Tomato___Spider_mites Two-spotted_spider_mite': {
                    "prevention": "Increase humidity and introduce natural predators.",
                    "organic": "Neem oil or insecticidal soap.",
                    "inorganic": "Use abamectin or bifenthrin cautiously."
                },
                'Tomato___Target_Spot': {
                    "prevention": "Remove lower leaves and improve ventilation.",
                    "organic": "Neem oil and compost tea spray.",
                    "inorganic": "Use azoxystrobin or mancozeb fungicide."
                },
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
                    "prevention": "Control whiteflies and remove infected plants.",
                    "organic": "Spray neem oil for whitefly control.",
                    "inorganic": "Apply imidacloprid insecticide."
                },
                'Tomato___Tomato_mosaic_virus': {
                    "prevention": "Use virus-free seeds and sterilize tools.",
                    "organic": "Use seaweed extract for plant immunity.",
                    "inorganic": "No direct cure; remove infected plants."
                },
                'Tomato___healthy': {
                    "prevention": "Maintain proper watering and balanced fertilizer.",
                    "organic": "Compost foliar spray as preventive.",
                    "inorganic": "No treatment required."
                }
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
    st.write("If the model prediction was incorrect, please select the correct disease name from the list below.")

    feedback_label = st.selectbox(
        "Select Correct Disease Name:",
        options=["-- Select Disease --"] + CLASS_NAMES
    )

    if st.button("Submit Feedback"):
        source_image = captured_image if captured_image else test_image
        if feedback_label == "-- Select Disease --":
            st.warning("⚠️ Please select a valid disease name before submitting.")
        elif source_image is None:
            st.warning("⚠️ No image selected or captured.")
        else:
            os.makedirs("feedback_data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{feedback_label}_{timestamp}.jpg"
            save_path = os.path.join("feedback_data", filename)

            img = Image.open(source_image)
            img.save(save_path)

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
