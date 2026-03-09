# Reflections

## Question 1

Why does transfer learning outperform training from scratch on this small dataset? Explain what features the Week 6 model learned that help with disaster classification.

Transfer learning starts from the Week 6 environmental model, which already learned low‑level and mid‑level visual features like edges, textures, shapes, horizons, buildings, vegetation, and water. Those features are also useful for recognizing disasters. Since our disaster dataset is small, a from‑scratch model does not see enough examples to learn good generic features and easily overfits or gets stuck with poor filters. The transfer model only has to learn a small new classifier head on top of rich, pre‑trained features, so it reaches higher accuracy much faster and more reliably.

## Question 2

Which disaster category was hardest to predict? Why do you think the model struggled with it?

Based on the confusion matrix, the hardest category tends to be the one with

 `(a) the fewest training examples and/or (b) visual overlap with other classes`

 For example, if "Blizzard" images share lots of white/gray textures with "Flood" or "Tornado" clouds, the model may confuse them because the underlying features (low contrast, cloudy sky, motion blur) look similar. Also, if that class appears in fewer images than the others, the model has less data to learn its unique patterns, which leads to more misclassifications.

## Question 3

In what real-world scenarios would transfer learning be the best approach? When would you train from scratch instead?

In a real-world senario transfer learning is best when training has been used for a similar task. For example, we have images of buildings, so training a model on urban traffic area's would save time as much of the data that would be collected from our building CCN.

## Question 4

 Which improvement strategy worked best? By how much did accuracy increase?

Baseline accuracy: 45%
Longer training accuracy: 65%
Fine-tuning accuracy: 35%

The best strategy was to use data augmentation via `ImageDataGenerator` using rotations, flips, and zoom. Longer training sessions with 20 epochs produced more accurate data. I also found that unfreezing the last few layers using `FINETUNE_LR = 1e-4`  helped with fine tuning. This improved accuracy by about ______ percentage points:  over the baseline. I think it helped because ________________________________.

## Question 5

Which strategy did not help as much as expected? Why do you think it was less effective?

Longer training started to overfit data, keeping epochs around 15-40 max produced the best results. An adujustment to fine-tuning was needed as too many layers with a high LR hurt the pre-trained models accuracy percentage.

## Question 6

What would you try next if you had more time and computing resources?

If I had more time and computing resources I would opt in to use a larger ImageNet model (VGG16 / ResNet50) as the base while adjusting the learning rate. With more data the augmentation is stronger as well so I would have trained on a larger dataset.

## Question 7

At what accuracy level would this model be useful for real disaster response?

Real disaster response might require very high recall for dangerous classes and false negatives such as missing a real disaster, are more dangerous than false positives. Wrong predictions could halt supply chains and miss-allocate resources for the disater occuring.
