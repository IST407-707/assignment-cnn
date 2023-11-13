# CNN Assignment

This is an **optional** assignment.  However, if you complete it, you can replace any one of your other assignment grades with the grade received here.

The objective of this assignment is to develop a convolutional neural network (CNN) using Keras to classify images from the CIFAR-10 dataset. CIFAR-10 is a widely used dataset in the machine learning community for benchmarking image recognition algorithms. It's a collection of images that are commonly used to train machine learning and computer vision algorithms. Here are some key points about CIFAR-10:

- The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.
- The classes include objects such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
- The dataset is divided into 50,000 training images and 10,000 test images.

Here's how to load CIFAR-10 using TensorFlow/Keras:

```python
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Optionally, normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0
```

In this code, `x_train` and `x_test` are arrays of image data (with images as 32x32 RGB pixel values), and `y_train` and `y_test` are arrays of labels (ranging from 0 to 9, corresponding to the 10 classes).

**Homework Tasks:**

1. **Data Exploration:**
   - Load the CIFAR-10 dataset.
   - Explore the dataset: visualize some images and their corresponding labels, check the shape of the training and test sets.

2. **Preprocessing:**
   - Normalize the image data.
   - Perform any additional preprocessing steps you find necessary (e.g., one-hot encoding of labels).

3. **Model Building:**
   - Design a CNN architecture using Keras. Suggest starting with a few convolutional layers, include dropout for regularization, and use pooling layers.
   - Add fully connected layers on top of the convolutional base.
   - Choose an appropriate activation function for the output layer (e.g., softmax for multi-class classification).

4. **Model Compilation:**
   - Compile the model with an appropriate loss function, optimizer, and metrics.

5. **Training:**
   - Train the model on the training data. Experiment with different batch sizes and number of epochs.

6. **Evaluation:**
   - Evaluate the model’s performance on the test data.
   - Report on metrics like accuracy or loss.

7. **Analysis:**
   - Analyze the results. Discuss what worked well and what could be improved.
   - Optionally, experiment with different model architectures, hyperparameters, or advanced techniques like data augmentation.

**Deliverables:**
A report or Jupyter notebook detailing the steps taken, including code, visualizations, and a discussion of the results and any conclusions drawn from the exercise.

### Getting Started with CNNs and the Sequential API

Here’s a basic outline of how you might construct a simple CNN for image classification using the Sequential API in Keras:

1. **Import Necessary Modules:**
   - Import the necessary components from Keras:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
     ```

2. **Initialize the Sequential Model:**
   - Create a new Sequential model object:
     ```python
     model = Sequential()
     ```

3. **Add Convolutional Layers:**
   - Add convolutional layers (`Conv2D`) with desired filters and kernel size. Typically, you'll include an activation function like ReLU.
     ```python
     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)))
     ```

4. **Add Pooling Layers:**
   - Follow each convolutional layer (or after a few conv layers) with a pooling layer (`MaxPooling2D`) to reduce spatial dimensions:
     ```python
     model.add(MaxPooling2D((2, 2)))
     ```

5. **Flatten the Output:**
   - After the last pooling layer, flatten the output to provide it to the dense layers:
     ```python
     model.add(Flatten())
     ```

6. **Add Fully Connected (Dense) Layers:**
   - Add one or more dense layers for classification. The last dense layer should have the number of neurons equal to the number of classes and typically uses the softmax activation function for multi-class classification:
     ```python
     model.add(Dense(64, activation='relu'))
     model.add(Dense(number_of_classes, activation='softmax'))
     ```

7. **Compile the Model:**
   - Compile the model with an optimizer, loss function, and metrics:
     ```python
     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     ```

8. **Train the Model:**
   - Fit the model to the training data:
     ```python
     model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
     ```

9. **Evaluate the Model:**
   - Evaluate the model on the test data:
     ```python
     test_loss, test_acc = model.evaluate(test_images, test_labels)
     ```
