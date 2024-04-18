# Industrial Equipment Classification

This project aims to classify images of industrial equipment into different categories using convolutional neural networks (CNNs) in TensorFlow. It includes data preprocessing, model training, evaluation, and manual testing functionalities.

Dataset = https://www.kaggle.com/datasets/endofnight17j03/industry-defect-dataset

## Features

- Data loading and preprocessing: Images and their corresponding labels are loaded and preprocessed using TensorFlow.
- CNN Architecture: The model architecture consists of convolutional layers followed by max-pooling layers, flattening layer, and dense layers for classification.
- Data Augmentation: ImageDataGenerator is used for data augmentation to improve model generalization.
- Model Training: The model is trained on the training data and evaluated on the test data.
- Evaluation Metrics: Accuracy, classification report, and confusion matrix are computed to evaluate model performance.
- Manual Testing: You can test the model manually by providing an image path.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Pandas
- Seaborn
- scikit-learn

## Example

To illustrate how to use the model for classification, two example images are provided (`example_equipment3.jpeg` and `example_equipment.jpeg`). You can use the `manual_test` function to predict the class of these images.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
