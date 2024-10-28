MNIST Digit Recognition using Convolutional Neural Networks (CNN)

Project Overview
This project implements a handwritten digit recognition system using Convolutional Neural Networks (CNN) on the MNIST dataset. The entire project is developed in a Jupyter Notebook (mnist_digit_recognition.ipynb), which provides an interactive environment for data exploration, model training, and evaluation.

Objectives
Build a CNN model to accurately classify handwritten digits (0-9).
Use PyTorch, a leading machine learning framework, to develop and train the model.
Provide visual insights into model performance, including correct and incorrect predictions.

Project Structure
mnist_digit_recognition.ipynb: The main Jupyter Notebook file containing all code, explanations, and visualizations related to the MNIST digit recognition task.

data/: Directory containing the MNIST dataset (automatically downloaded if not present).

Methodology
Data Preprocessing:

The MNIST dataset is loaded and transformed into tensors. Images are normalized to a range of [-1, 1] for effective training.

Data loaders are created for batch processing of training and testing datasets.

Model Architecture:

A CNN architecture consisting of two convolutional layers, max-pooling layers, and fully connected layers is implemented.

The model uses ReLU activation functions to introduce non-linearity.

Training the Model:

The model is trained using the Adam optimizer with Cross-Entropy loss function for classification tasks.
The training loop consists of multiple epochs where the model learns to minimize the loss through backpropagation.

Evaluation:

The model's performance is evaluated on the test dataset.
The notebook provides visualizations of predictions to showcase the model's accuracy.

Custom Image Prediction:

The project includes functionality to load and predict digits from custom handwritten images.
Users can test the model's effectiveness on their own images directly within the notebook.

Key Features
Interactive Jupyter Notebook: The entire project is contained within a Jupyter Notebook, allowing for easy modification and experimentation.

High Accuracy: The model achieves significant accuracy on the MNIST test dataset.

Visualization: The notebook includes plots and visuals to demonstrate model predictions and performance.

Custom Image Support: Users can input their own handwritten digits for classification within the notebook.

Tools and Technologies

Programming Language: Python
Framework: PyTorch
Libraries:
torch for model development and training.
torchvision for dataset handling and image transformations.
matplotlib for visualizations.
PIL (Python Imaging Library) for image processing.
Getting Started

To run this project:

Clone the Repository: Use the command below to clone the project repository.


git clone <repository-url>
cd <repository-directory>

Install Required Libraries: Ensure you have the necessary libraries installed. You can install them using pip:

pip install torch torchvision matplotlib

Open the Jupyter Notebook: Launch Jupyter Notebook in the project directory.

jupyter notebook mnist_digit_recognition.ipynb

Run the Notebook: Execute the cells in the notebook to train the model and make predictions.

Conclusion
The MNIST Digit Recognition project showcases the effectiveness of CNNs in image classification tasks within an interactive Jupyter Notebook. By utilizing deep learning techniques with the PyTorch framework, the model learns to classify handwritten digits accurately. This project serves as an excellent introduction to deep learning and computer vision for both beginners and experienced practitioners.