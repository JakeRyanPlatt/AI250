"""
AI 250 - Week 7 Assignment: Improving Transfer Learning Performance
Student: Jacob Platt

"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# REPRODUCIBILITY
# -------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DATA_DIR = "Training Images"                  # Disaster images folder
WEEK6_MODEL = "environmental_classifier_model_true.h5"  # Week 6 transfer base

IMG_SIZE = 150
BATCH_SIZE = 8

# Baseline vs improvement settings
EPOCHS_BASELINE = 5        # Baseline: short training
EPOCHS_LONGER = 20         # Strategy 2: train longer
EPOCHS_FINETUNE = 10       # Strategy 3: unfreeze last layers and fine-tune
FINETUNE_LR = 1e-4         # Lower LR for fine-tuning

print("=" * 60)
print("WEEK 7: IMPROVING TRANSFER LEARNING PERFORMANCE")
print("=" * 60)

# -------------------------------------------------------------------------
# STEP 1: LOAD DATA WITH DATA AUGMENTATION (Strategy 1)
# -------------------------------------------------------------------------
print("\n📂 Loading disaster dataset with data augmentation...")

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode="nearest",
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"\n✓ Dataset loaded: {num_classes} classes")
print("  Classes:", ", ".join(class_names))
print(f"  Training samples:   {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")

# -------------------------------------------------------------------------
# STEP 2: HELPER TO BUILD TRANSFER MODEL
# -------------------------------------------------------------------------
def build_transfer_model(finetune=False, finetune_lr=1e-4):
    """
    Load Week 6 model, remove its final classification layer, and attach a new
    disaster classifier head.

    If finetune=False:
        - Freeze all base layers (classic transfer learning)

    If finetune=True:
        - Freeze most layers but unfreeze the last few for fine-tuning
        - Use a lower learning rate
    """
    if not os.path.exists(WEEK6_MODEL):
        raise FileNotFoundError(
            f"Week 6 model not found at '{WEEK6_MODEL}'. "
            "Make sure environmental_classifier_model.h5 is in this folder."
        )

    base_model = keras.models.load_model(WEEK6_MODEL)

    # Remove the original top classification layer
    base_without_top = models.Sequential(base_model.layers[:-1])

    # Freeze / unfreeze layers on the base model
    if not finetune:
        for layer in base_without_top.layers:
            layer.trainable = False
        optimizer = keras.optimizers.Adam()
    else:
        for layer in base_without_top.layers[:-3]:
            layer.trainable = False
        for layer in base_without_top.layers[-3:]:
            layer.trainable = True
        optimizer = keras.optimizers.Adam(learning_rate=finetune_lr)

    # Stack new head on top of the frozen / partially frozen base
    model = models.Sequential([
        base_without_top,
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name="disaster_classifier"),
    ])

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -------------------------------------------------------------------------
# STEP 3: BASELINE TRANSFER MODEL (FROZEN BASE, FEW EPOCHS)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE: TRANSFER LEARNING (FROZEN BASE, 5 EPOCHS)")
print("=" * 60)

baseline_model = build_transfer_model(finetune=False)
baseline_model.summary()

print(f"\n🏋️ Training baseline model for {EPOCHS_BASELINE} epochs...")
history_baseline = baseline_model.fit(
    train_generator,
    epochs=EPOCHS_BASELINE,
    validation_data=val_generator,
    verbose=1,
)

val_loss_base, val_acc_base = baseline_model.evaluate(val_generator, verbose=0)
print(f"\n✅ Baseline Validation Accuracy: {val_acc_base:.1%}")

# -------------------------------------------------------------------------
# STEP 4: IMPROVEMENT STRATEGY 2 - TRAIN LONGER
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("IMPROVEMENT 1: TRAIN LONGER (FROZEN BASE, 20 EPOCHS)")
print("=" * 60)

longer_model = build_transfer_model(finetune=False)
print(f"\n🏋️ Training longer model for {EPOCHS_LONGER} epochs...")
history_longer = longer_model.fit(
    train_generator,
    epochs=EPOCHS_LONGER,
    validation_data=val_generator,
    verbose=1,
)

val_loss_long, val_acc_long = longer_model.evaluate(val_generator, verbose=0)
print(f"\n✅ Longer-Training Validation Accuracy: {val_acc_long:.1%}")

# -------------------------------------------------------------------------
# STEP 5: IMPROVEMENT STRATEGY 3 - UNFREEZE LAST LAYERS (FINE-TUNE)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("IMPROVEMENT 2: UNFREEZE LAST LAYERS + LOW LR (FINE-TUNE)")
print("=" * 60)

finetune_model = build_transfer_model(finetune=True, finetune_lr=FINETUNE_LR)
finetune_model.summary()

print(f"\n🏋️ Training fine-tuned model for {EPOCHS_FINETUNE} epochs...")
history_finetune = finetune_model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_generator,
    verbose=1,
)

val_loss_ft, val_acc_ft = finetune_model.evaluate(val_generator, verbose=0)
print(f"\n✅ Fine-Tuned Validation Accuracy: {val_acc_ft:.1%}")

# -------------------------------------------------------------------------
# STEP 6: COMPARISON VISUALIZATION (BAR PLOT OF ACCURACIES)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON: BASELINE VS IMPROVED MODELS")
print("=" * 60)

print(f"Baseline (5 epochs, frozen):      {val_acc_base:.1%}")
print(f"Train longer (20 epochs, frozen): {val_acc_long:.1%}")
print(f"Fine-tune (unfreeze last layers): {val_acc_ft:.1%}")

labels = ["Baseline", "Longer", "Fine-tune"]
accs = [val_acc_base, val_acc_long, val_acc_ft]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accs, color=["gray", "skyblue", "orange"])
plt.ylim(0, 1.0)
plt.ylabel("Validation Accuracy")
plt.title("Transfer Learning: Baseline vs Improved Models")

for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.02,
        f"{acc:.1%}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("transfer_improvement_comparison.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: transfer_improvement_comparison.png")

# -------------------------------------------------------------------------
# STEP 7: CONFUSION MATRIX & CLASSIFICATION REPORT (BEST MODEL)
# -------------------------------------------------------------------------
best_model = finetune_model
best_name = "Fine-tuned Model"
best_acc = val_acc_ft

if val_acc_long > best_acc:
    best_model = longer_model
    best_name = "Longer-Training Model"
    best_acc = val_acc_long
if val_acc_base > best_acc:
    best_model = baseline_model
    best_name = "Baseline Model"
    best_acc = val_acc_base

print(f"\nBest model for confusion matrix: {best_name} ({best_acc:.1%} accuracy)")

val_generator.reset()
y_pred_probs = best_model.predict(val_generator, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "Count"},
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{best_name} - Confusion Matrix")
plt.tight_layout()
plt.savefig("best_model_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("✓ Saved: best_model_confusion_matrix.png")

print("\n📋 Classification Report (best model):")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# -------------------------------------------------------------------------
# STEP 8: SAVE MODELS (OPTIONAL BUT NICE)
# -------------------------------------------------------------------------
baseline_model.save("disaster_transfer_baseline.h5")
longer_model.save("disaster_transfer_longer.h5")
finetune_model.save("disaster_transfer_finetuned.h5")
print("\n✓ Saved models: disaster_transfer_baseline.h5, "
      "disaster_transfer_longer.h5, disaster_transfer_finetuned.h5")

print("\n" + "=" * 60)
print("ASSIGNMENT RUN COMPLETE")
print("=" * 60)


