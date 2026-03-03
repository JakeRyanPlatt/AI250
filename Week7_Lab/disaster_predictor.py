"""
AI 250 - Week 7: Disaster Image Predictor
Interactive demo: Predict disaster type from any image

Usage:
  python disaster_predictor.py path/to/image.jpg
  python disaster_predictor.py --demo

Author: Axis (OpenClaw)
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'disaster_transfer_model.h5'
IMG_SIZE = 150
CLASS_NAMES = ['Blizzard', 'Earthquake', 'Fire', 'Flood', 'Tornado']  # Alphabetical order

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model():
    """Load the trained disaster detection model"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Please run disaster_transfer_demo.py first to train the model.")
        sys.exit(1)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!\n")
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)

def predict_disaster(model, image_path, show_plot=True):
    """Predict disaster type from image"""
    
    print(f"Analyzing image: {image_path}")
    print("-" * 60)
    
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    # Print results
    print(f"\n🎯 PREDICTION: {predicted_class}")
    print(f"   Confidence: {confidence:.1f}%\n")
    
    print("📊 All Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[i] * 100
        bar = "█" * int(prob / 2)  # Visual bar (50% = 25 chars)
        print(f"   {class_name:12s} {prob:5.1f}% {bar}")
    
    # Visualize if requested
    if show_plot:
        visualize_prediction(original_img, predicted_class, confidence, predictions)
    
    return predicted_class, confidence, predictions

def visualize_prediction(img, predicted_class, confidence, predictions):
    """Display image with prediction overlay"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f"Predicted: {predicted_class} ({confidence:.1f}%)", 
                  fontsize=14, fontweight='bold')
    
    # Show probability bar chart
    colors = ['#2ecc71' if i == np.argmax(predictions) else '#95a5a6' 
              for i in range(len(CLASS_NAMES))]
    bars = ax2.barh(CLASS_NAMES, predictions * 100, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=11)
    ax2.set_title('Disaster Type Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def demo_mode(model):
    """Run predictions on sample images from training set"""
    
    print("\n" + "=" * 60)
    print("DEMO MODE: Testing on sample images")
    print("=" * 60 + "\n")
    
    # Find sample images from each category
    training_dir = "Training Images"
    
    if not os.path.exists(training_dir):
        print("❌ Training Images folder not found!")
        return
    
    # Get one sample from each category
    for category in CLASS_NAMES:
        category_path = os.path.join(training_dir, category)
        if os.path.exists(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Pick first image
                sample_image = os.path.join(category_path, images[0])
                print(f"\n{'='*60}")
                print(f"Testing on actual {category} image:")
                print(f"{'='*60}")
                predict_disaster(model, sample_image, show_plot=False)
                print()

def interactive_mode(model):
    """Interactive mode: keep predicting until user quits"""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter image paths to predict disaster type.")
    print("Type 'quit' to exit.\n")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not image_path:
            continue
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}\n")
            continue
        
        print()
        predict_disaster(model, image_path, show_plot=True)
        print("\n" + "-" * 60 + "\n")

def main():
    """Main function"""
    
    print("=" * 60)
    print("DISASTER IMAGE PREDICTOR")
    print("=" * 60)
    print()
    
    # Load model
    model = load_model()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--demo':
            # Demo mode: test on training samples
            demo_mode(model)
        
        elif arg == '--interactive' or arg == '-i':
            # Interactive mode
            interactive_mode(model)
        
        elif os.path.exists(arg):
            # Single image prediction
            predict_disaster(model, arg, show_plot=True)
        
        else:
            print(f"❌ Invalid argument or file not found: {arg}")
            print("\nUsage:")
            print("  python disaster_predictor.py path/to/image.jpg")
            print("  python disaster_predictor.py --demo")
            print("  python disaster_predictor.py --interactive")
            sys.exit(1)
    
    else:
        # No arguments: show usage and enter interactive mode
        print("Usage:")
        print("  python disaster_predictor.py <image_path>    Predict single image")
        print("  python disaster_predictor.py --demo          Test on training samples")
        print("  python disaster_predictor.py --interactive   Interactive mode")
        print()
        
        # Ask user what to do
        choice = input("Enter mode (1=single image, 2=demo, 3=interactive): ").strip()
        
        if choice == '1':
            image_path = input("Image path: ").strip()
            if os.path.exists(image_path):
                predict_disaster(model, image_path, show_plot=True)
            else:
                print(f"❌ File not found: {image_path}")
        
        elif choice == '2':
            demo_mode(model)
        
        elif choice == '3':
            interactive_mode(model)
        
        else:
            print("Invalid choice!")
            sys.exit(1)

if __name__ == "__main__":
    main()
