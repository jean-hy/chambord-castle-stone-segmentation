# Development of a dataset for training and testing deep learning algorithms for stone segmentation in the Château de Chambord.

This project aims to develop a robust dataset and train state-of-the-art deep learning models for stone segmentation on the facades of the Château de Chambord. By accurately isolating stone elements, this work contributes to the preservation of historical monuments through automated flaw detection.

![Chambord Castle](https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Chambord_Castle_Northwest_facade.jpg/2880px-Chambord_Castle_Northwest_facade.jpg)

[Source: Wikipedia](https://fr.wikipedia.org/wiki/Ch%C3%A2teau_de_Chambord)

## Project Overview

- **Dataset Creation:**  
  Use the Segment Anything Model 2 (SAM 2) to segment images of the Château de Chambord’s facades, followed by cleaning the results to keep only stone elements. The cleaned images are then divided into 256x256 patches.

- **Data Augmentation:**  
  Increase dataset diversity using techniques such as rotation, flipping, zooming, and brightness adjustments.

- **Deep Learning Models:**  
  Train and evaluate multiple deep learning models (a Convolutional Neural Network and Transformer-based architectures) for accurate stone segmentation.

- **Results Comparison:**  
  Compare model performance (using metrics like accuracy, IoU, precision, and recall) against benchmarks from published research.
