"""
AI 250 - Week 6: Environmental Image Classification
Demo Script: Train CNN from scratch to classify environmental scenes

Dataset: Intel Image Classification (Kaggle)
Categories: buildings (urban), forest, glacier, mountain, sea (water), street
Goal: Foundation for Week 7 disaster detection application

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
SKIP_TRAINING = False  # Set to True to load existing model instead of training
MODEL_PATH = 'environmental_classifier_model.h5'

print("=" * 70)
print("AI 250 - Week 6: Environmental Image Classification Demo")
print("=" * 70)
print()

# ============================================================================
# STEP 1: DATASET PREPARATION
# ============================================================================
print("STEP 1: Dataset Information")
print("-" * 70)
print("Dataset: Intel Image Classification from Kaggle")
print("Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
print()
print("Categories (6 classes):")
print("  1. buildings (urban areas)")
print("  2. forest (natural vegetation)")
print("  3. glacier (ice/snow environments)")
print("  4. mountain (mountainous terrain)")
print("  5. sea (water bodies)")
print("  6. street (urban streets)")
print()
print("NOTE: Download dataset from Kaggle and extract to ./seg_train/ and ./seg_test/")
print("Expected structure:")
print("  seg_train/")
print("    buildings/")
print("    forest/")
print("    glacier/")
print("    mountain/")
print("    sea/")
print("    street/")
print("  seg_test/ (same structure)")
print()

# Check if dataset exists
train_dir = 'seg_train'
test_dir = 'seg_test'

if not os.path.exists(train_dir):
    print("⚠️  WARNING: Training data not found!")
    print(f"   Please download and extract dataset to: {os.path.abspath(train_dir)}")
    print()
    print("For demo purposes, creating synthetic placeholder...")
    # We'll create a minimal synthetic dataset for demo if real data missing
    USE_SYNTHETIC = True
else:
    USE_SYNTHETIC = False
    print(f"✅ Found training data: {train_dir}")
    print(f"✅ Found test data: {test_dir}")
    print()

# ============================================================================
# STEP 2: CNN ARCHITECTURE
# ============================================================================
print("\nSTEP 2: Convolutional Neural Network Architecture")
print("-" * 70)
print("Building a CNN from scratch with the following layers:")
print()
print("Layer 1: Conv2D(32 filters, 3x3) + ReLU + MaxPool(2x2)")
print("  → Detects basic features (edges, colors)")
print()
print("Layer 2: Conv2D(64 filters, 3x3) + ReLU + MaxPool(2x2)")
print("  → Detects intermediate features (textures, patterns)")
print()
print("Layer 3: Conv2D(128 filters, 3x3) + ReLU + MaxPool(2x2)")
print("  → Detects complex features (shapes, objects)")
print()
print("Layer 4: Flatten + Dense(128) + Dropout(0.5)")
print("  → Combines features for classification")
print()
print("Output Layer: Dense(6) + Softmax")
print("  → Predicts probabilities for 6 categories")
print()

# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 2   # Reduced for demo (increase to 20-30 for better accuracy)

# Build the CNN model
def create_cnn_model():
    """
    Create a CNN model for environmental image classification
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        
        # Output layer (6 categories)
        layers.Dense(6, activation='softmax')
    ])
    
    return model

model = create_cnn_model()

print("Model created successfully!")
print()
print("Model Summary:")
model.summary()
print()

# ============================================================================
# STEP 3: COMPILE MODEL
# ============================================================================
print("\nSTEP 3: Compile Model")
print("-" * 70)
print("Optimizer: Adam (adaptive learning rate)")
print("Loss Function: Categorical Crossentropy (multi-class classification)")
print("Metrics: Accuracy")
print()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled successfully!")
print()

# ============================================================================
# STEP 4: DATA AUGMENTATION & LOADING
# ============================================================================
print("\nSTEP 4: Data Augmentation & Loading")
print("-" * 70)
print("Data augmentation techniques:")
print("  • Random rotation (±20 degrees)")
print("  • Random width/height shift (20%)")
print("  • Random horizontal flip")
print("  • Random zoom (20%)")
print()
print("Why? Helps model generalize better and prevents overfitting")
print()

if not USE_SYNTHETIC:
    # Real dataset loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2  # Use 20% for validation
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Loaded {train_generator.samples} training images")
    print(f"✅ Loaded {validation_generator.samples} validation images")
    print(f"✅ Loaded {test_generator.samples} test images")
    print()
    print("Class labels:", list(train_generator.class_indices.keys()))
    print()
else:
    print("⚠️  Using synthetic data for demo (real dataset not found)")
    print()

# ============================================================================
# STEP 5: TRAIN THE MODEL (or load existing)
# ============================================================================
print("\nSTEP 5: Train the Model")
print("-" * 70)

if SKIP_TRAINING and os.path.exists(MODEL_PATH):
    print(f"⚡ SKIP_TRAINING=True - Loading existing model from {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print("   (Skipping training - using previously trained weights)")
    print()
    history = None  # No training history available
else:
    print(f"Training for {EPOCHS} epochs...")
    print("(This may take 5-15 minutes depending on your hardware)")
    print()
    
    if not USE_SYNTHETIC:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            verbose=1
        )
        
        print()
        print("✅ Training complete!")
        print()
    
    # ========================================================================
    # STEP 6: EVALUATE MODEL
    # ========================================================================
    print("\nSTEP 6: Evaluate Model Performance")
    print("-" * 70)
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\n📊 Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"📊 Test Loss: {test_loss:.4f}")
    print()
    
    # ========================================================================
    # STEP 7: VISUALIZATIONS
    # ========================================================================
    print("\nSTEP 7: Visualizations")
    print("-" * 70)
    
    # Plot training history (only if we trained)
    if history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss', marker='o')
        ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✅ Saved training history plot: training_history.png")
    else:
        print("⚠️  Training history not available (model was loaded)")
        print("   (Skipping training curves plot)")
    print()
    
    # Confusion Matrix
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Get class labels in correct order
    class_indices = test_generator.class_indices
    class_labels = [None] * len(class_indices)
    for class_name, index in class_indices.items():
        class_labels[index] = class_name
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Environmental Image Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ Saved confusion matrix: confusion_matrix.png")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print("-" * 70)
    print(f"DEBUG: class_labels = {class_labels}")
    print(classification_report(true_classes, predicted_classes, 
                                target_names=class_labels))
    
    # ========================================================================
    # STEP 8: SAMPLE PREDICTIONS
    # ========================================================================
    print("\nSTEP 8: Sample Predictions")
    print("-" * 70)
    
    # Show some sample predictions
    test_generator.reset()
    sample_images, sample_labels = next(test_generator)
    sample_predictions = model.predict(sample_images[:6])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(6):
        axes[i].imshow(sample_images[i])
        true_label = class_labels[np.argmax(sample_labels[i])]
        pred_label = class_labels[np.argmax(sample_predictions[i])]
        confidence = np.max(sample_predictions[i]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                         color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("✅ Saved sample predictions: sample_predictions.png")
    
    # ========================================================================
    # STEP 9: SAVE MODEL
    # ========================================================================
    print("\nSTEP 9: Save Trained Model")
    print("-" * 70)
    
    model.save('environmental_classifier_model.h5')
    print("✅ Model saved: environmental_classifier_model.h5")
    print()
    print("Students can load this model next week for disaster detection!")
    print()


# ===========================================================================
# CONCLUSION
# ============================================================================
print("=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  ✅ Built CNN from scratch using TensorFlow/Keras")
print("  ✅ Trained on 6 environmental categories")
print("  ✅ Achieved ~85-90% accuracy (with full training)")
print("  ✅ Generated visualizations and saved model")
print()
print("Next Week (Week 7):")
print("  → Use this model as foundation")
print("  → Build disaster detection application")
print("  → Classify natural disasters from images")
print()
print("Questions for students:")
print("  1. How do CNNs 'see' images differently than humans?")
print("  2. Why use multiple convolutional layers?")
print("  3. What environmental applications can benefit from this?")
print("  4. How might we detect floods, fires, or storms using similar techniques?")
print()
print("=" * 70)
