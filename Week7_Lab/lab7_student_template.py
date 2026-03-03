"""
AI 250 - Week 7: Transfer Learning Lab
Jacob Platt

YOUR MISSION: Compare transfer learning vs training from scratch!

You'll build TWO models:
1. Transfer Learning: Reuse your Week 6 nature classifier
2. From Scratch: Train a new CNN on disaster data only

Then compare which approach works better with small datasets.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_DIR = "Training Images"
WEEK6_MODEL = "environmental_classifier_model.h5"
IMG_SIZE = 150
BATCH_SIZE = 16
EPOCHS = 5

print("=" * 60)
print("WEEK 7: TRANSFER LEARNING LAB")
print("=" * 60)

# ============================================================================
# STEP 1: Load the Disaster Dataset
# ============================================================================
print("\n📂 STEP 1: Loading disaster dataset...")

#  Create an ImageDataGenerator with:

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create training generator from DATA_DIR
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
    shuffle=False
)

# Create validation generator

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
    shuffle=False
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"✓ Dataset loaded: {num_classes} classes")
print(f"  Classes: {', '.join(class_names)}")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")

# ============================================================================
# STEP 2: Transfer Learning (Reuse Week 6 Model)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: TRANSFER LEARNING")


print("=" * 60)


# Check if week 6 model exists
if not os.path.exists(WEEK6_MODEL):
    print(f"\n⚠️  ERROR: Week 6 model not found at {WEEK6_MODEL}")
    print("   Please run Week 6 lab first to create the model!")
    exit(1)

print(f"Loading Week 6 model from: {WEEK6_MODEL}")
base_model = keras.models.load_model(Week6_MODEL)

print("\nOriginal Week 6 model structure:")
base_model.summary()


print("\n🔧 Removing top layer...")
base_model_without_top = models.Sequential(base_model.layers[:-1])


print("🔒 Freezing convolutional layers...")
for layer in base_model_without_top.layers:
    layers.trainable = False

# TODO: Create transfer model by adding new classification layer
#   Hint: models.Sequential([base_model_without_top, layers.Dense(...)])
#   The Dense layer should have num_classes units and 'softmax' activation

print("➕ Adding new disaster classification layer...")
transfer_model = models.Sequential([
    base_model_without_top,
    layers.Dense(num_classes, activation='softmax', name='disaster_classifier')

])

#  Compile the transfer model
transfer_model.compile(
    optimizer='adam'
    loss='categorical_crossentropy'
    metrics=['accuracy']
)

print("\nTransfer model summary:")
transfer_model.summary()

# Count trainable parameters
trainable_params = sum([tf.size(w).numpy() for w in transfer_model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in transfer_model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(f"\n📊 Parameter counts:")
print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"   Frozen: {non_trainable_params:,} ({non_trainable_params/total_params*100:.1f}%)")

print(f"\nTraining transfer model ({EPOCHS_TRANSFER} epochs)...")
history_transfer = transfer_model.fit(
    train_datagen,
    epochs=EPOCHS_TRANSFER,
    validation_data=val_data,
    verbose=1
)
history_scratch = scratch_model.fit(
    train_datagen,
    validation_data=val_data,
    epochs=EPOCHS,
)

print(f"\n🏋️  Training transfer model ({EPOCHS} epochs)...")
history_transfer = None  # YOUR CODE HERE

# Evaluate
val_loss_transfer, val_acc_transfer = transfer_model.evaluate(val_generator)
print(f"\n✅ Transfer Learning Validation Accuracy: {val_acc_transfer:.1%}")

# ============================================================================
# STEP 3: Train From Scratch
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: TRAIN FROM SCRATCH")
print("=" * 60)

# Build a CNN from scratch
print("\nBuilding a new CNN (no pre-trained weights...)...")
scratch_model= model.Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3))
       MaxPooling2D((2,2))
       Conv2D(64, (3,3), activation='relu')
       MaxPooling2D((2,2))
       Conv2D(64, (3,3), activation='relu')
       MaxPooling2D((2,2))
       Flatten()
       Dense(64, activation='relu')
       Dense(num_classes, activation='softmax')
])


# TODO: Compile the scratch model (same settings as transfer model)

scratch_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nFrom scratch model summary:")
scratch_model.summary()

# TODO: Train the from-scratch model

print(f"\n🏋️  Training from scratch ({EPOCHS} epochs)...")
history_scratch = scratch_model.fit(
    train_generator,
    epochs=EPOCHS_SCRATCH,
    validation_data=val_generator,
    verbose=1
)

# Evaluate
val_loss_scratch, val_acc_scratch = scratch_model.evaluate(val_generator)
print(f"\n✅ From Scratch Validation Accuracy: {val_acc_scratch:.1%}")

# ============================================================================
# STEP 4: Compare Results
# ============================================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\n📊 Results:")
print(f"   Transfer Learning: {val_acc_transfer:.1%}")
print(f"   From Scratch:      {val_acc_scratch:.1%}")

# Calculate improvement percentage


improvement = (val_acc_transfer - val_acc_scratch) / val_acc_scratch * 100

print(f"\n🎯 Transfer Learning is {improvement:+.1f}% better!")

print("\n💡 REFLECTION QUESTIONS:")
print("   1. Why does transfer learning perform better?")
print("   2. What features from nature images help with disasters?")
print("   3. When would training from scratch be better?")

# ============================================================================
# STEP 5: Visualization
# ============================================================================
print("\n📈 Creating visualizations...")

# TODO: Create a figure with 2 subplots (1 row, 2 columns)
#   Subplot 1: Accuracy curves (both models)
#   Subplot 2: Loss curves (both models)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy curves
axes[0].plot(history_transfer.history['accuracy'], label='Transfer (Train)', linewidth=2)
axes[0].plot(history_transfer.history['val_accuracy'], label='Transfer (Val)', linewidth=2)
axes[0].plot(history_scratch.history['accuracy'], label='Scratch (Train)', linewidth=2, linestyle='--')
axes[0].plot(history_scratch.history['val_accuracy'], label='Scratch (Val)', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training Accuracy: Transfer vs From Scratch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss curves
axes[1].plot(history_transfer.history['loss'], label='Transfer (Train)', linewidth=2)
axes[1].plot(history_transfer.history['val_loss'], label='Transfer (Val)', linewidth=2)
axes[1].plot(history_scratch.history['loss'], label='Scratch (Train)', linewidth=2, linestyle='--')
axes[1].plot(history_scratch.history['val_loss'], label='Scratch (Val)', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss: Transfer vs From Scratch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_curves.png', dpi=150, bbox_inches='tight')
print("✓ Saved: comparison_curves.png")

print("\n📊 Creating confusion matrix...")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Transfer Learning Model - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('transfer_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: transfer_confusion_matrix.png")


plt.savefig('confusion_matrix.png', dpi=150)
print("✓ Saved: confusion_matrix.png")


#   classification_report(y_true, y_pred_classes, target_names=class_names)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Save the best model
transfer_model.save('disaster_transfer_model.h5')
print("\n✓ Saved: disaster_transfer_model.h5")

print("\n" + "=" * 60)
print("LAB COMPLETE! 🎉")
print("=" * 60)
print("\nSubmit:")
print("  1. This completed script")
print("  2. comparison_curves.png")
print("  3. confusion_matrix.png")
print("  4. Your answers to the reflection questions")
