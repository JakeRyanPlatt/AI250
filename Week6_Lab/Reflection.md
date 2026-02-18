Reflection Questions (discuss with your group)
----------------------------------------------------------------------

1. How does your model 'see' an image differently than you do?
   (Hint: Think about the convolutional layers)

Our model uses convolutional layers to process images, which is a key difference from how humans perceive images.
The convolutional layers allow the model to scan the image in small patches and extract features such as edges,
lines, and shapes. This process can be seen as a form of "visual abstraction" where the model is trying to identify
patterns and relationships in the image that are not immediately apparent to humans.

In contrast, when we look at an image, our brains use a more holistic approach, taking into account the context and
meaning of the image as a whole. We also use higher-level cognitive processes such as attention, memory, and
experience to interpret the visual information.

1. Why did we use data augmentation?
   (Hint: What happens if we only show the model one angle/zoom?)

We used data augmentation to increase the size of our dataset and improve the model's robustness to variations in
images. By applying random transformations such as rotation, scaling, and flipping to the original images, we can
create new training examples that are similar yet distinct from the original ones.

If we only showed the model one angle or zoom, it would likely be less effective at recognizing objects and patterns
in the image. Data augmentation helps the model learn to generalize across different variations of an image, making
it more accurate and reliable.

1. What does the confusion matrix tell you about your model?
   (Hint: Which categories does it confuse most often?)

The confusion matrix provides a detailed breakdown of how well our model is performing on each class or category. It
tells us which classes are most confusing for the model, where it tends to misclassify certain examples as belonging
to other categories.

For example, if we look at the precision and recall values for each class in the confusion matrix, we can see which
classes are causing problems for the model. This information can be used to adjust our training strategy, such as by
adding more data or tweaking the hyperparameters of the model.

1. How could this model be applied to environmental monitoring?
   (Hint: Think deforestation, glacial melting, urbanization...)

This model could be applied to environmental monitoring in a variety of ways, including:

* Deforestation detection - By training the model on images of forests and comparing them to satellite imagery, we
can detect areas where trees are being cleared or removed.
* Glacial melting monitoring - The model can be used to analyze images of glaciers and track changes in their size
and shape over time.
* Urbanization monitoring - The model can help identify areas where urbanization is occurring and assess the impact
on local ecosystems.

In each case, the model can provide a more objective and accurate assessment than human interpretation alone,
helping scientists and policymakers make informed decisions about environmental protection and conservation.

1. What improvements could make this model more accurate?
   (Hint: More data? Different architecture? Longer training?)

Based on our analysis of the confusion matrix, it seems that the model is struggling to recognize certain classes or
patterns in the data. To improve accuracy, we might consider:

* Collecting more data: Adding more examples of each class to the training set can help the model learn to recognize
them more accurately.
* Adjusting the architecture: Changing the configuration of the convolutional layers or adding new layers could help
the model learn to recognize patterns and relationships in the data that are not currently being captured.
* Increasing training time: Longer training times can allow the model to converge on better solutions and improve
its accuracy.
======================================================================
