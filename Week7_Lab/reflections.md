# Reflections

## Question 1

Why does transfer learning outperform training from scratch on this small dataset? Explain what features the Week 6 model learned that help with disaster classification.

Transfer learning starts from the Week 6 environmental model, which already learned low‑level and mid‑level visual features like edges, textures, shapes, horizons, buildings, vegetation, and water. Those features are also useful for recognizing disasters (e.g., smoke texture for fires, water patterns for floods, cloud structures for storms). Because our disaster dataset is small, a from‑scratch model does not see enough examples to learn good generic features and easily overfits or gets stuck with poor filters. The transfer model only has to learn a small new classifier head on top of rich, pre‑trained features, so it reaches higher accuracy much faster and more reliably.

## Question 2

Which disaster category was hardest to predict? Why do you think the model struggled with it?

Based on the confusion matrix, the hardest category tends to be the one with
 (a) the fewest training examples and/or (b) visual overlap with other classes.
 For example, if "Blizzard" images share lots of white/gray textures with
 "Flood" or "Tornado" clouds, the model may confuse them because the underlying features (low contrast, cloudy sky, motion blur) look similar. Also, if that class appears in fewer images than the others, the model has less data to learn its unique patterns, which leads to more misclassifications.

## Question 3

In what real-world scenarios would transfer learning be the best approach? When would you train from scratch instead?
