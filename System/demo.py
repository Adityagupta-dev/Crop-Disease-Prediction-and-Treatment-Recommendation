import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Define paths - Update these for your environment
TOMATO_MODEL_PATH = 'models/Tomato_EfficientNetB0_finetuned_final.h5'
WATERMELON_MODEL_PATH = 'models/Watermelon_MobileNetV2_final.h5'
TEST_IMAGES_PATH = 'test_images/'

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

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for model prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to load image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        img = img / 255.0
        return img, np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def predict_disease(model, preprocessed_image, class_names):
    """Predict disease class for an image"""
    prediction = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100
    
    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_predictions = [(class_names[i], prediction[0][i] * 100) for i in top_indices]
    
    return predicted_class, confidence, top_predictions

def get_treatment_recommendation(predicted_class, crop_type):
    """Get treatment recommendation based on predicted disease"""
    if crop_type == 'tomato':
        return Config.TOMATO_TREATMENTS.get(predicted_class)
    else:  # watermelon
        return Config.WATERMELON_TREATMENTS.get(predicted_class)

def display_results(image, predicted_class, confidence, top_predictions, treatment_info):
    """Display the prediction results and treatment recommendations"""
    plt.figure(figsize=(14, 10))
    
    # Display image with prediction
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f"Detected: {predicted_class.replace('_', ' ')}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    
    # Display probability chart
    plt.subplot(2, 2, 2)
    bars = plt.bar(
        [p[0].replace('_', ' ') for p in top_predictions],
        [p[1] for p in top_predictions],
        color=['green', 'orange', 'red']
    )
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    plt.title('Disease Probability Distribution')
    plt.ylabel('Probability (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Display treatment recommendations
    plt.subplot(2, 1, 2)
    treatment_text = (
        f"TREATMENT RECOMMENDATIONS\n\n"
        f"Detected Disease: {predicted_class.replace('_', ' ')}\n\n"
        f"Recommended Product: {treatment_info['product']}\n\n"
        f"Application Instructions: {treatment_info['instructions']}\n\n"
        f"Prevention Measures: {treatment_info['prevention']}"
    )
    
    plt.text(0.1, 0.5, treatment_text, wrap=True, fontsize=12, 
             verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def demo_prediction(model_path, image_path, crop_type):
    """Demonstrate the prediction pipeline for a single image"""
    print(f"\nProcessing {crop_type} image: {image_path}")
    
    # Load model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and preprocess image
    original_img, preprocessed_img = load_and_preprocess_image(image_path)
    if original_img is None or preprocessed_img is None:
        return
    print("Image preprocessed successfully")
    
    # Make prediction
    if crop_type == 'tomato':
        class_names = Config.TOMATO_CLASSES
    else:  # watermelon
        class_names = Config.WATERMELON_CLASSES
    
    predicted_class, confidence, top_predictions = predict_disease(model, preprocessed_img, class_names)
    print(f"Prediction complete: {predicted_class} with {confidence:.2f}% confidence")
    
    # Get treatment recommendation
    treatment_info = get_treatment_recommendation(predicted_class, crop_type)
    
    # Display results
    display_results(original_img, predicted_class, confidence, top_predictions, treatment_info)
    
    return predicted_class, confidence, treatment_info

def detect_crop_type(image_path=None):
    """Simple command-line interface to select crop type"""
    if image_path is None:
        # If no image path provided, ask for one
        image_path = input("Enter the full path to your image: ")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None, None
    
    # Ask for crop type if not obvious from filename
    print("\nSelect crop type:")
    print("1. Tomato")
    print("2. Watermelon")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        crop_type = 'tomato'
        model_path = TOMATO_MODEL_PATH
    elif choice == '2':
        crop_type = 'watermelon'
        model_path = WATERMELON_MODEL_PATH
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return None, None, None
        
    return image_path, crop_type, model_path

# Main execution code
if __name__ == "__main__":
    # Disable eager execution warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    print("Plant Disease Detection System")
    print("------------------------------")
    
    # Option 1: Use command line arguments to directly process a specific image
    import sys
    if len(sys.argv) > 1:
        # If specific path is provided as command line argument
        test_image_path = sys.argv[1]
        
        # Check if crop type is also provided
        if len(sys.argv) > 2 and sys.argv[2].lower() in ['tomato', 'watermelon']:
            crop_type = sys.argv[2].lower()
            model_path = TOMATO_MODEL_PATH if crop_type == 'tomato' else WATERMELON_MODEL_PATH
            
            # Process the image
            demo_prediction(model_path, test_image_path, crop_type)
        else:
            # If only image path provided, ask for crop type
            image_path, crop_type, model_path = detect_crop_type(test_image_path)
            if image_path:
                demo_prediction(model_path, image_path, crop_type)
    else:
        # Option 2: Interactive menu
        while True:
            print("\nSelect an option:")
            print("1. Test specific watermelon image")
            print("2. Test specific tomato image")
            print("3. Choose your own image and crop type")
            print("4. Exit program")
            
            choice = input("Enter your choice (1, 2, 3, or 4): ")
            
            if choice == '1':
                # Test watermelon image
                test_image_path = input("Enter path to watermelon image: ")  # FIXED: Changed prompt text
                if os.path.exists(test_image_path):
                    demo_prediction(WATERMELON_MODEL_PATH, test_image_path, 'watermelon')
                else:
                    print(f"Error: Watermelon image not found at {test_image_path}")
                    print("Please provide the correct path.")
            
            elif choice == '2':
                # Ask for tomato image path
                test_image_path = input("Enter path to tomato image: ")
                if os.path.exists(test_image_path):
                    demo_prediction(TOMATO_MODEL_PATH, test_image_path, 'tomato')
                else:
                    print(f"Error: Tomato image not found at {test_image_path}")
                    print("Please provide the correct path.")
                    
            elif choice == '3':
                # Choose custom image and crop type
                image_path, crop_type, model_path = detect_crop_type()
                if image_path:
                    demo_prediction(model_path, image_path, crop_type)
            
            elif choice == '4':
                print("Exiting program. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")