import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import random
import time
from tensorflow.keras.mixed_precision import set_global_policy

# Check for GPU availability and set up
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Using GPU")
    # Allow memory growth to prevent TF from allocating all GPU memory at once
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # Enable mixed precision training for better performance on GPU
    set_global_policy('mixed_float16')
else:
    print("No GPU found, using CPU")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuration
class Config:
    # General settings
    IMG_SIZE = 224
    BATCH_SIZE = 64  # Increased batch size for GPU
    EPOCHS = 15
    INITIAL_EPOCHS = 10  # For initial training before fine-tuning
    FINE_TUNE_EPOCHS = 5  # For fine-tuning phase
    LEARNING_RATE = 0.001
    
    # Paths - Update these based on your directory structure
    TOMATO_DATASET_PATH = r"c:\Users\ACER\OneDrive\Documents\Desktop\tomato\train"
    WATERMELON_DATASET_PATH = r"C:\Users\ACER\OneDrive\Documents\Desktop\Augmented_Image"
    MODEL_SAVE_PATH = 'models/'
    
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
    
    # Rallis product recommendations for tomato diseases
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
    
    # Rallis product recommendations for watermelon diseases
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

# Data Preparation

def create_data_generators():
    """Create data generators for training and validation data with optimized settings"""
    
    # Create data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,  # Plants have natural orientation
        fill_mode='nearest',
        validation_split=0.2  # 20% of data will be used for validation
    )
    
    # Create a separate generator for validation with only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen, test_datagen

def load_tomato_data(train_datagen, val_datagen):
    """Load and prepare tomato dataset with optimized settings"""
    
    # Use tf.data.Dataset for better performance
    train_generator = train_datagen.flow_from_directory(
        Config.TOMATO_DATASET_PATH,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        Config.TOMATO_DATASET_PATH,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def load_watermelon_data(train_datagen, val_datagen):
    """Load and prepare watermelon dataset with optimized settings"""
    
    train_generator = train_datagen.flow_from_directory(
        Config.WATERMELON_DATASET_PATH,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        Config.WATERMELON_DATASET_PATH,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

# Model Building

def create_model(num_classes, model_name='EfficientNetB0'):
    """Create a transfer learning model using efficient architectures"""
    
    if model_name == 'EfficientNetB0':
        # EfficientNet is better for smaller GPUs like the RTX 3050
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    else:
        raise ValueError("Model name must be either 'EfficientNetB0' or 'MobileNetV2'")
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers - simplified for faster training
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)  # Increased dropout for better generalization
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model with mixed precision
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator, model_name, epochs=None):
    """Train the model with callbacks and optimized settings"""
    
    if epochs is None:
        epochs = Config.INITIAL_EPOCHS
    
    # Create necessary directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        f"{Config.MODEL_SAVE_PATH}/{model_name}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,  # Reduced patience for faster adaptation
        min_lr=1e-6,
        verbose=1
    )
    
    # Calculate steps correctly to avoid validation issues
    steps_per_epoch = train_generator.samples // Config.BATCH_SIZE
    validation_steps = validation_generator.samples // Config.BATCH_SIZE
    
    # Ensure at least 1 step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Save the final model
    model.save(f"{Config.MODEL_SAVE_PATH}/{model_name}_final.h5")
    
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator, model_name):
    """Fine-tune model by unfreezing layers with optimized approach"""
    
    print(f"Fine-tuning the {model_name} model...")
    
    # Unfreeze the top layers of the base model
    if 'EfficientNetB0' in model_name:
        # Unfreeze the last 30% of layers
        unfreeze_layers = int(len(base_model.layers) * 0.3)
        for layer in base_model.layers[-unfreeze_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    elif 'MobileNetV2' in model_name:
        # Unfreeze the last 20 layers
        for layer in base_model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with remaining epochs
    fine_tune_history = train_model(
        model, 
        train_generator, 
        validation_generator, 
        f"{model_name}_finetuned", 
        epochs=Config.FINE_TUNE_EPOCHS
    )
    
    return fine_tune_history

def evaluate_model(model, validation_generator, class_names):
    """Evaluate model performance and display metrics"""
    
    # Prepare data for evaluation
    validation_steps = validation_generator.samples // Config.BATCH_SIZE
    validation_steps = max(1, validation_steps)  # Ensure at least 1 step
    
    # Predict using the model
    print("Evaluating model performance...")
    predictions = model.predict(validation_generator, steps=validation_steps)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = validation_generator.classes[:len(predicted_classes)]
    
    # Generate classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_SAVE_PATH}/confusion_matrix.png")
    
    return report_df, cm

def visualize_training_history(history, model_name):
    """Visualize training history"""
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_SAVE_PATH}/{model_name}_training_history.png")

# Prediction and Treatment Recommendation

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(model, image_path, class_names):
    """Predict disease class for a single image"""
    
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100
    
    return predicted_class, confidence

def get_treatment_recommendation(predicted_class, crop_type):
    """Get treatment recommendation based on predicted disease"""
    
    if crop_type == 'tomato':
        return Config.TOMATO_TREATMENTS.get(predicted_class, {
            'product': 'No specific recommendation available',
            'instructions': 'Consult with an agricultural expert.',
            'prevention': 'Follow general disease prevention practices.'
        })
    elif crop_type == 'watermelon':
        return Config.WATERMELON_TREATMENTS.get(predicted_class, {
            'product': 'No specific recommendation available',
            'instructions': 'Consult with an agricultural expert.',
            'prevention': 'Follow general disease prevention practices.'
        })
    else:
        return {
            'product': 'Invalid crop type specified',
            'instructions': 'Please specify either "tomato" or "watermelon".',
            'prevention': 'N/A'
        }

def display_prediction_with_treatment(image_path, predicted_class, confidence, treatment_info):
    """Display the prediction results and treatment recommendations"""
    
    # Read and display the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    
    # Display image with prediction
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    
    # Display treatment recommendations
    plt.subplot(1, 2, 2)
    recommendation_text = (
        f"TREATMENT RECOMMENDATIONS\n\n"
        f"Detected Disease: {predicted_class}\n\n"
        f"Recommended Product: {treatment_info['product']}\n\n"
        f"Application Instructions: {treatment_info['instructions']}\n\n"
        f"Prevention Measures: {treatment_info['prevention']}"
    )
    
    plt.text(0.1, 0.5, recommendation_text, wrap=True, fontsize=12, 
             verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_SAVE_PATH}/prediction_result.png")
    plt.show()

# Optimized training process

def train_tomato_model():
    """Train and evaluate the tomato disease model with optimized settings"""
    
    print("Starting Tomato Disease Detection Model Training...")
    start_time = time.time()
    
    # Create data generators
    train_datagen, val_datagen, _ = create_data_generators()
    
    # Load data
    train_generator, validation_generator = load_tomato_data(train_datagen, val_datagen)
    
    # Create model - using EfficientNetB0 as it's more efficient for smaller GPUs
    model_name = 'Tomato_EfficientNetB0'
    tomato_model, base_model = create_model(len(Config.TOMATO_CLASSES), 'EfficientNetB0')
    
    # Print model summary
    print(f"Model: {model_name}")
    tomato_model.summary()
    
    # Train model
    history = train_model(tomato_model, train_generator, validation_generator, model_name)
    
    # Visualize training
    visualize_training_history(history, model_name)
    
    # Fine-tune model
    fine_tune_history = fine_tune_model(tomato_model, base_model, train_generator, validation_generator, model_name)
    
    # Visualize fine-tuning
    visualize_training_history(fine_tune_history, f"{model_name}_finetuned")
    
    # Evaluate model
    report, cm = evaluate_model(tomato_model, validation_generator, Config.TOMATO_CLASSES)
    
    print("Tomato Disease Classification Report:")
    print(report)
    
    total_time = time.time() - start_time
    print(f"Total tomato model training time: {total_time:.2f} seconds")
    
    return tomato_model

def train_watermelon_model():
    """Train and evaluate the watermelon disease model with optimized settings"""
    
    print("Starting Watermelon Disease Detection Model Training...")
    start_time = time.time()
    
    # Create data generators
    train_datagen, val_datagen, _ = create_data_generators()
    
    # Load data
    train_generator, validation_generator = load_watermelon_data(train_datagen, val_datagen)
    
    # Create model - using MobileNetV2 as it's efficient and the watermelon dataset is simpler
    model_name = 'Watermelon_MobileNetV2'
    watermelon_model, base_model = create_model(len(Config.WATERMELON_CLASSES), 'MobileNetV2')
    
    # Print model summary
    print(f"Model: {model_name}")
    watermelon_model.summary()
    
    # Train model
    history = train_model(watermelon_model, train_generator, validation_generator, model_name)
    
    # Visualize training
    visualize_training_history(history, model_name)
    
    # Fine-tune model
    fine_tune_history = fine_tune_model(watermelon_model, base_model, train_generator, validation_generator, model_name)
    
    # Visualize fine-tuning
    visualize_training_history(fine_tune_history, f"{model_name}_finetuned")
    
    # Evaluate model
    report, cm = evaluate_model(watermelon_model, validation_generator, Config.WATERMELON_CLASSES)
    
    print("Watermelon Disease Classification Report:")
    print(report)
    
    total_time = time.time() - start_time
    print(f"Total watermelon model training time: {total_time:.2f} seconds")
    
    return watermelon_model

def inference_demo(tomato_model, watermelon_model, image_path, crop_type):
    """Demo function to show prediction and treatment recommendation"""
    
    if crop_type.lower() == 'tomato':
        predicted_class, confidence = predict_disease(tomato_model, image_path, Config.TOMATO_CLASSES)
    elif crop_type.lower() == 'watermelon':
        predicted_class, confidence = predict_disease(watermelon_model, image_path, Config.WATERMELON_CLASSES)
    else:
        print(f"Invalid crop type: {crop_type}. Please specify 'tomato' or 'watermelon'.")
        return
    
    treatment_info = get_treatment_recommendation(predicted_class, crop_type.lower())
    display_prediction_with_treatment(image_path, predicted_class, confidence, treatment_info)

# Enhanced main execution function with timing
def run_training_pipeline():
    """Run the complete training pipeline with timing"""
    
    print("=" * 50)
    print("CROP DISEASE DETECTION SYSTEM")
    print("GPU-Optimized Training Pipeline")
    print("=" * 50)
    
    total_start_time = time.time()
    
    # Create directory for model saving
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Train tomato model
    print("\n" + "=" * 50)
    print("TRAINING TOMATO DISEASE MODEL")
    print("=" * 50)
    tomato_model = train_tomato_model()
    
    # Train watermelon model
    print("\n" + "=" * 50)
    print("TRAINING WATERMELON DISEASE MODEL")
    print("=" * 50)
    watermelon_model = train_watermelon_model()
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"TOTAL TRAINING TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 50)
    
    # Example inference (uncomment and modify path to test)
    # inference_demo(tomato_model, watermelon_model, 'path/to/test_image.jpg', 'tomato')
    
    return tomato_model, watermelon_model

# If this script is run directly, execute training and sample inference
if __name__ == "__main__":
    run_training_pipeline()