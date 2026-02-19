axes = axes.ravel()
for i in range(6):
axes[i].imshow(sample_images[i])
true_label = class_labels[np.argmax(sample_labels[i])]
pred_label = class_labels[np.argmax(sample_predictions[i])]
confidence = np.max(sample_predictions[i]) * 100
color = 'green' if true_label == pred_label else 'red'
axes[i].set_title(f'True: {true_label}\nPred: {pred_label}
({confidence:.1f}%)',
color=color, fontweight='bold')
axes[i].axis('off')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
print(" Saved sample predictions: sample_predictions.png")✅
# ========================================================================
# STEP 9: SAVE MODEL
# ========================================================================
print("\nSTEP 9: Save Trained Model")
print("-" * 70)
model.save('environmental_classifier_model.h5')
print(" Model saved: environmental_classifier_model.h5")✅
print()
print("Students can load this model next week for disaster detection!")
print()
# ============================================================================
# CONCLUSION
# ============================================================================
print("=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print(" Built CNN from scratch using TensorFlow/Keras")✅
print(" Trained on 6 environmental categories")✅
print(" Achieved ~85-90% accuracy (with full training)")✅
print(" Generated visualizations and saved model")✅
print()
print("Next Week (Week 7):")
print(" → Use this model as foundation")
print(" → Build disaster detection application")
print(" → Classify natural disasters from images")
print()
print("Questions for students:")
print(" 1. How do CNNs 'see' images differently than humans?")
print(" 2. Why use multiple convolutional layers?")
print(" 3. What environmental applications can benefit from this?")
print(" 4. How might we detect floods, fires, or storms using similar
techniques?")
print()
print("=" * 70)
