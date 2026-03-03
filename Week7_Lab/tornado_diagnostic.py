"""
Quick diagnostic: Check what the model predicts for each tornado image
"""

import os
import numpy as np
from tensorflow import keras
from PIL import Image

MODEL_PATH = 'disaster_transfer_model.h5'
IMG_SIZE = 150
CLASS_NAMES = ['Blizzard', 'Earthquake', 'Fire', 'Flood', 'Tornado']

# Load model
model = keras.models.load_model(MODEL_PATH)

# Check tornado images
tornado_dir = "Training Images/Tornado"
tornado_images = sorted([f for f in os.listdir(tornado_dir) if f.endswith('.jpg')])

print(f"Analyzing {len(tornado_images)} tornado images...")
print("=" * 70)

confusion_count = {cls: 0 for cls in CLASS_NAMES}
correct = 0

for img_file in tornado_images:
    img_path = os.path.join(tornado_dir, img_file)
    
    # Load and preprocess
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    
    # Track confusion
    confusion_count[predicted_class] += 1
    if predicted_class == 'Tornado':
        correct += 1
        marker = "✓"
    else:
        marker = "✗"
    
    print(f"{marker} {img_file:15s} → {predicted_class:12s} ({confidence:5.1f}%)")

print("\n" + "=" * 70)
print(f"Tornado Accuracy: {correct}/{len(tornado_images)} = {correct/len(tornado_images)*100:.1f}%")
print("\nConfusion breakdown:")
for cls, count in sorted(confusion_count.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  Predicted as {cls:12s}: {count:2d} images ({count/len(tornado_images)*100:.1f}%)")
