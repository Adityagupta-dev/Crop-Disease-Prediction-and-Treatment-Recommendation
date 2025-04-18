import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import streamlit as st
from PIL import Image
import io

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define paths - Update these for your environment
TOMATO_MODEL_PATH = 'models/Tomato_EfficientNetB0_finetuned_final.h5'
WATERMELON_MODEL_PATH = 'models/Watermelon_MobileNetV2_final.h5'
CONFUSION_MATRIX_PATH = 'models/confusion_matrix.png'
WATERMELON_TRAINING_HISTORY = 'models/Watermelon_MobileNetV2_training_history.png'
TOMATO_TRAINING_HISTORY = 'models/Tomato_EfficientNetB0_finetuned_training_history.png'

# Configuration
class Config:
    # Image size
    IMG_SIZE = 224
    
    # Class mappings
    TOMATO_CLASSES = [
        'Tomato_mosaic_virus',
        'Target_Spot',
        'Bacterial_spot',
        'Tomato_Yellow_Leaf_Curl_Virus',
        'Late_blight',
        'Leaf_Mold',
        'Early_blight',
        'Spider_mites',  # Two-spotted_spider_mite
        'Septoria_leaf_spot',
        'Healthy'
    ]
    
    WATERMELON_CLASSES = [
        'Anthracnose',
        'Downy_Mildew',
        'Mosaic_Virus',
        'Healthy'
    ]
    
    # Treatments dictionary
    TOMATO_TREATMENTS = {
        'Tomato_mosaic_virus': {
            'product': 'No specific Rallis chemical treatment available',
            'instructions': 'Remove infected plants, control insect vectors, and implement sanitation practices.',
            'prevention': 'Use resistant varieties and implement strict sanitation.'
        },
        'Target_Spot': {
            'product': 'Rallis Taqat (Mancozeb 75% WP)',
            'instructions': 'Apply 2.0-2.5g/L of water. Spray at 10-15 day intervals.',
            'prevention': 'Maintain good air circulation and avoid overhead irrigation.'
        },
        'Bacterial_spot': {
            'product': 'Rallis Blitox (Copper Oxychloride 50% WP)',
            'instructions': 'Apply 2.5-3.0g/L of water every 7-10 days.',
            'prevention': 'Use disease-free seeds and practice crop rotation.'
        },
        'Tomato_Yellow_Leaf_Curl_Virus': {
            'product': 'Rallis Virtako (Thiamethoxam + Chlorantraniliprole)',
            'instructions': 'Apply 0.4g/L to control whitefly vectors.',
            'prevention': 'Use reflective mulches to repel whiteflies and remove host weeds.'
        },
        'Late_blight': {
            'product': 'Rallis Master (Metalaxyl 8% + Mancozeb 64% WP)',
            'instructions': 'Apply 2.5g/L of water every 7 days during favorable conditions.',
            'prevention': 'Plant resistant varieties and ensure good drainage.'
        },
        'Leaf_Mold': {
            'product': 'Rallis Saaf (Carbendazim 12% + Mancozeb 63% WP)',
            'instructions': 'Apply 2.0g/L of water every 10 days.',
            'prevention': 'Reduce humidity and improve air circulation.'
        },
        'Early_blight': {
            'product': 'Rallis Companion (Carbendazim 12% + Mancozeb 63% WP)',
            'instructions': 'Apply 2.0-2.5g/L of water every 10-15 days.',
            'prevention': 'Maintain proper plant spacing and practice mulching.'
        },
        'Spider_mites': {
            'product': 'Rallis Roket (Fenpyroximate 5% EC)',
            'instructions': 'Apply 1ml/L of water. Ensure coverage on leaf undersides.',
            'prevention': 'Maintain adequate soil moisture and control dust on plants.'
        },
        'Septoria_leaf_spot': {
            'product': 'Rallis Ergon (Kresoxim-methyl 44.3% SC)',
            'instructions': 'Apply 0.5-0.75ml/L of water every 10-14 days.',
            'prevention': 'Remove infected leaves and avoid overhead irrigation.'
        },
        'Healthy': {
            'product': 'No treatment needed',
            'instructions': 'Continue regular maintenance and monitoring.',
            'prevention': 'Follow good agricultural practices for disease prevention.'
        }
    }
    
    WATERMELON_TREATMENTS = {
        'Anthracnose': {
            'product': 'Rallis Captaf (Captan 50% WP)',
            'instructions': 'Apply 2.5g/L of water every 7-10 days during humid conditions.',
            'prevention': 'Use disease-free seeds and practice crop rotation.'
        },
        'Downy_Mildew': {
            'product': 'Rallis Master (Metalaxyl 8% + Mancozeb 64% WP)',
            'instructions': 'Apply 2.5g/L of water every 7-10 days as preventive measure.',
            'prevention': 'Maintain good air circulation and avoid overhead irrigation.'
        },
        'Mosaic_Virus': {
            'product': 'Rallis Virtako (Thiamethoxam + Chlorantraniliprole)',
            'instructions': 'Apply 0.4g/L to control aphid vectors.',
            'prevention': 'Control weed hosts and aphid populations.'
        },
        'Healthy': {
            'product': 'No treatment needed',
            'instructions': 'Continue regular maintenance and monitoring.',
            'prevention': 'Implement proper sanitation and crop rotation.'
        }
    }

# Function to preprocess image from upload
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model prediction"""
    try:
        # Read image as bytes
        image_bytes = uploaded_file.getvalue()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Failed to load the image. Please try again with a different image.")
            return None, None
            
        # Convert to RGB (from BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to expected size
        img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # Expand dimensions for model input
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        return img, img_expanded
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# Function to make predictions
def predict_disease(model, preprocessed_image, class_names):
    """Predict disease class for an image"""
    try:
        prediction = model.predict(preprocessed_image)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = [(class_names[i], prediction[0][i] * 100) for i in top_indices]
        
        return predicted_class, confidence, top_predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Function to get treatment recommendation
def get_treatment_recommendation(predicted_class, crop_type):
    """Get treatment recommendation based on predicted disease"""
    if crop_type == 'tomato':
        return Config.TOMATO_TREATMENTS.get(predicted_class)
    else:  # watermelon
        return Config.WATERMELON_TREATMENTS.get(predicted_class)

# Function to display prediction results
def display_results(image, predicted_class, confidence, top_predictions, treatment_info):
    """Display the prediction results and treatment recommendations"""
    # Display original image
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)
    
    # Create columns for results and chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Result")
        st.write(f"**Detected Disease:** {predicted_class.replace('_', ' ')}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    
    with col2:
        # Disease probability chart
        st.subheader("Probability Distribution")
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot data
        labels = [p[0].replace('_', ' ') for p in top_predictions]
        values = [p[1] for p in top_predictions]
        colors = ['green', 'orange', 'red'] if len(top_predictions) == 3 else ['green', 'red']
        
        bars = ax.bar(labels, values, color=colors)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
    
    # Treatment Recommendations
    st.subheader("Treatment Recommendations")
    
    st.markdown(f"""
    | Category | Recommendation |
    | --- | --- |
    | **Recommended Product** | {treatment_info['product']} |
    | **Application Instructions** | {treatment_info['instructions']} |
    | **Prevention Measures** | {treatment_info['prevention']} |
    """)

# Streamlit Homepage
def home_page():
    st.title("Plant Disease Detection System")
    
    # Add a banner image
    st.image("https://via.placeholder.com/800x300?text=Plant+Disease+Detection+System", use_column_width=True)
    
    st.markdown("""
    ## Welcome to the Plant Disease Detection Tool
    
    This application helps farmers and gardeners identify plant diseases in tomatoes and watermelons using artificial intelligence.
    
    ### Features:
    - **Fast Disease Detection**: Upload an image and get instant results
    - **Disease Information**: Learn about the detected disease
    - **Treatment Recommendations**: Get specific treatment guidance
    - **Prevention Tips**: Learn how to prevent future occurrences
    
    ### Supported Crops:
    - **Tomato**: 10 different classes including 9 diseases and healthy plants
    - **Watermelon**: 4 different classes including 3 diseases and healthy plants
    
    ### How to Use:
    1. Navigate to the "Disease Detection" page
    2. Upload an image of your plant leaf
    3. Select the crop type
    4. Get instant analysis and treatment recommendations
    
    """)
    
# Disease Detection Page
def detection_page():
    st.title("Plant Disease Detection")
    
    st.write("Upload a photo of your plant leaf to detect diseases and get treatment recommendations")
    
    # Crop selection
    crop_type = st.radio("Select Plant Type:", ("tomato", "watermelon"))
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display a spinner while processing
        with st.spinner("Processing image..."):
            # Preprocess the image
            original_img, preprocessed_img = preprocess_image(uploaded_file)
            
            if original_img is not None and preprocessed_img is not None:
                # Load appropriate model
                try:
                    model_path = TOMATO_MODEL_PATH if crop_type == 'tomato' else WATERMELON_MODEL_PATH
                    model = load_model(model_path)
                    
                    # Get class names based on crop type
                    class_names = Config.TOMATO_CLASSES if crop_type == 'tomato' else Config.WATERMELON_CLASSES
                    
                    # Make prediction
                    predicted_class, confidence, top_predictions = predict_disease(model, preprocessed_img, class_names)
                    
                    # Get treatment recommendation
                    treatment_info = get_treatment_recommendation(predicted_class, crop_type)
                    
                    # Display results
                    display_results(original_img, predicted_class, confidence, top_predictions, treatment_info)
                    
                except Exception as e:
                    st.error(f"Error loading model or making prediction: {str(e)}")
        with col2:
            st.markdown("""
            ### Instructions:
            1. Select the crop type (tomato or watermelon)
            2. Upload a clear image of the plant leaf
            3. Make sure the leaf is well-lit and focused
            4. Wait for the analysis to complete
            """)

# About the Model Page
def about_model_page():
    st.title("About the Models")
    
    st.markdown("""
    ## Model Architecture and Training
    
    This system uses two deep learning models for disease detection:
    
    ### Tomato Disease Model
    - **Architecture**: EfficientNetB0 (fine-tuned)
    - **Classes**: 10 classes (9 diseases + healthy)
    - **Training Dataset**: Over 18,000 images of tomato leaves
    - **Accuracy**: ~97% on test set
    """)
    
    # Display tomato model training history if available
    try:
        st.image(TOMATO_TRAINING_HISTORY, caption="Tomato Model Training History", use_column_width=True)
    except:
        st.warning("Tomato model training history image not found.")
    
    st.markdown("""
    ### Watermelon Disease Model
    - **Architecture**: MobileNetV2
    - **Classes**: 4 classes (3 diseases + healthy)
    - **Training Dataset**: Over 5,000 images of watermelon leaves
    - **Accuracy**: ~95% on test set
    """)
    
    # Display watermelon model training history if available  
    try:
        st.image(WATERMELON_TRAINING_HISTORY, caption="Watermelon Model Training History", use_column_width=True)
    except:
        st.warning("Watermelon model training history image not found.")
    
    st.markdown("---")
    
    st.markdown("""
    ## Model Performance
    
    Both models were evaluated using confusion matrices and various metrics including precision, recall, and F1-score.
    """)
    
    # Display confusion matrix if available
    try:
        st.image(CONFUSION_MATRIX_PATH, caption="Model Confusion Matrix", use_column_width=True)
    except:
        st.warning("Confusion matrix image not found.")
    
    st.markdown("""
    ## Disease Classes
    
    ### Tomato Diseases:
    - Tomato Mosaic Virus
    - Target Spot
    - Bacterial Spot
    - Tomato Yellow Leaf Curl Virus
    - Late Blight
    - Leaf Mold
    - Early Blight
    - Spider Mites
    - Septoria Leaf Spot
    - Healthy
    
    ### Watermelon Diseases:
    - Anthracnose
    - Downy Mildew
    - Mosaic Virus
    - Healthy
    """)

# Help and Guide Page
def help_guide_page():
    st.title("Help & Guide")
    
    st.markdown("""
    ## How to Use This Application
    
    ### Getting Started
    1. Navigate to the "Disease Detection" page using the sidebar menu
    2. Select the crop type (tomato or watermelon)
    3. Upload a clear image of the plant leaf
    4. Review the detection results and treatment recommendations
    
    ### Best Practices for Image Capture
    - Take photos in good natural light
    - Make sure the leaf is in focus and clearly visible
    - Include the entire leaf in the frame
    - Avoid shadows or glare on the leaf surface
    
    ### Understanding Results
    - **Disease Name**: The detected plant disease
    - **Confidence**: How confident the model is in its prediction
    - **Probability Distribution**: Shows top predicted diseases and their likelihood
    - **Treatment Recommendations**: Suggested products and application instructions
    - **Prevention Measures**: Tips to prevent future occurrences
    
    ### Common Issues
    
    | Problem | Solution |
    | --- | --- |
    | Low confidence score | Take another photo with better lighting and focus |
    | Model won't load | Check your internet connection and refresh the page |
    | Incorrect prediction | Try multiple photos from different angles |
    
    ## About Disease Detection
    
    Plant disease detection uses computer vision and deep learning to identify plant diseases from images. This technology helps farmers detect diseases early, reducing crop losses and pesticide use.
    """)
    
    st.markdown("---")
    
    st.subheader("Contact Information")
    st.markdown("""
    For support, feature requests, or collaboration:
    
    - **LinkedIn**: [Aditya Gupta](https://www.linkedin.com/in/aditya-gupta-062478250)
    - **Email**: contact@example.com (example placeholder)
    """)
    
# Main app
def main():
    st.set_page_config(
        page_title="Plant Disease Detection System",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Create navigation
    page = st.sidebar.radio("Go to", ["Home", "Disease Detection", "About the Models", "Help & Guide"])
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Plant Disease Detection System**  
    Developed for farmers and agricultural specialists
    
    Version 1.0
    """)
    
    # Display appropriate page based on selection
    if page == "Home":
        home_page()
    elif page == "Disease Detection":
        detection_page()
    elif page == "About the Models":
        about_model_page()
    elif page == "Help & Guide":
        help_guide_page()

if __name__ == "__main__":
    main()
