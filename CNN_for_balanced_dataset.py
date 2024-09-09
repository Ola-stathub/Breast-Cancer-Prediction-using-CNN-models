#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import shutil
import zipfile
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D



import os

import shutil


# In[ ]:


get_ipython().system('kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset')


# In[ ]:


get_ipython().system('unzip breast-ultrasound-images-dataset.zip -d dataset_dir')


# In[ ]:


zip_path = '/content/breast-ultrasound-images-dataset.zip'
extract_to = '/content/breast-ultrasound-images-dataset'  # where to extract the files

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)


# In[ ]:


# Check the contents of the extracted directory
dataset_dir = extract_to
print(os.listdir(dataset_dir))


# In[ ]:


dataset_dir = '/content/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT'  # Path to the directory containing benign, malignant, and normal folders


# In[ ]:


# Set paths
malignant_dir = os.path.join(dataset_dir, 'malignant')
benign_dir = os.path.join(dataset_dir, 'benign')
normal_dir = os.path.join(dataset_dir, 'normal')

# Create directories for the split data
base_dir = "/content/split_dataset"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Create subdirectories for each class
for category in ['malignant', 'benign', 'normal']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)


# In[ ]:


##Function to upsample the minority class


# In[ ]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def augment_images(class_dir, target_count):
    current_count = len(os.listdir(class_dir))
    images_to_generate = target_count - current_count

    # If current_count is already greater than or equal to target_count, do nothing
    if images_to_generate <= 0:
        print(f"No augmentation needed for {class_dir}. Current count: {current_count}")
        return

    # Load all images from the class directory
    image_files = os.listdir(class_dir)
    generated_count = 0

    while generated_count < images_to_generate:
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = load_img(img_path)
            x = img_to_array(img)  # Convert image to numpy array
            x = np.expand_dims(x, axis=0)  # Reshape to (1, width, height, channels)

            # Generate augmented images
            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='jpeg'):
                generated_count += 1
                if generated_count >= images_to_generate:
                    break
        if generated_count >= images_to_generate:
            break

    print(f"Finished augmenting {class_dir}. Total images: {len(os.listdir(class_dir))}")

# Target number of images
target_count = 884

# Upsample Malignant and Normal classes to have the same number of images as Benign
augment_images(malignant_dir, target_count)
augment_images(normal_dir, target_count)


# In[ ]:


# Recalculate the number of images in each class
num_malignant = len(os.listdir(malignant_dir))
num_benign = len(os.listdir(benign_dir))
num_normal = len(os.listdir(normal_dir))

print(f'Number of Malignant images: {num_malignant}')
print(f'Number of Benign images: {num_benign}')
print(f'Number of Normal images: {num_normal}')


# Prepare data for the bar chart
classes = ['Malignant', 'Benign', 'Normal']
frequencies = [num_malignant, num_benign, num_normal]

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(classes, frequencies, color=['olivedrab', 'olive', 'slategrey'])

# Add labels and title
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Frequency of Images in Each Class')
plt.show()


# In[ ]:


# Function to split data
def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_SIZE):
    data = []
    for item in os.listdir(SOURCE):
        item_path = os.path.join(SOURCE, item)
        if os.path.getsize(item_path) > 0:
            data.append(item)

    random.shuffle(data)
    train_size = int(SPLIT_SIZE * len(data))
    validation_size = int((len(data) - train_size) / 2)

    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    test_data = data[train_size + validation_size:]

    for item in train_data:
        shutil.copy(os.path.join(SOURCE, item), os.path.join(TRAINING, item))

    for item in validation_data:
        shutil.copy(os.path.join(SOURCE, item), os.path.join(VALIDATION, item))

    for item in test_data:
        shutil.copy(os.path.join(SOURCE, item), os.path.join(TESTING, item))

# Define split size (e.g., 70% train, 15% validation, 15% test)
split_size = 0.7

# Split the data
split_data(malignant_dir, os.path.join(train_dir, 'malignant'), os.path.join(validation_dir, 'malignant'), os.path.join(test_dir, 'malignant'), split_size)
split_data(benign_dir, os.path.join(train_dir, 'benign'), os.path.join(validation_dir, 'benign'), os.path.join(test_dir, 'benign'), split_size)
split_data(normal_dir, os.path.join(train_dir, 'normal'), os.path.join(validation_dir, 'normal'), os.path.join(test_dir, 'normal'), split_size)


# In[ ]:


#Fitting my proposed CNN model


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 3  # Benign, Malignant, Normal

# Initialize the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the CNN
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset
train_generator = train_datagen.flow_from_directory('/content/split_dataset/train',
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('/content/split_dataset/validation',
                                                        target_size=(img_width, img_height),
                                                        batch_size=32,
                                                        class_mode='categorical')

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // validation_generator.batch_size,
                    epochs=25)

# Save the model
#model.save('breast_cancer_classifier.h5')


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


##LeNet-5 pretrained model


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Image size for LeNet-5
img_width, img_height = 32, 32

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values between 0 and 1
    rotation_range=10,             # Randomly rotate images by up to 10 degrees
    width_shift_range=0.1,         # Randomly shift images horizontally
    height_shift_range=0.1,        # Randomly shift images vertically
    shear_range=0.1,               # Shear transformation
    zoom_range=0.1,                # Zoom in/out by 10%
    horizontal_flip=True,          # Randomly flip images horizontally
    validation_split=0.2)          # Split data into training and validation sets

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data

# Train set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Training data

# Validation set
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Validation data

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

# Defining LeNet-5 model
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(img_width, img_height, 1)),
    AveragePooling2D(pool_size=(2, 2)),  # Add pool_size argument
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),  # Add pool_size argument
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: malignant, benign, normal
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator)


# In[ ]:


# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[ ]:


#Alex Net Pre-trained model


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size for AlexNet
img_width, img_height = 227, 227

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values between 0 and 1
    rotation_range=20,             # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,         # Randomly shift images horizontally
    height_shift_range=0.2,        # Randomly shift images vertically
    shear_range=0.2,               # Shear transformation
    zoom_range=0.2,                # Zoom in/out by 20%
    horizontal_flip=True,          # Randomly flip images horizontally
    validation_split=0.2)          # Split data into training and validation sets

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data

# Train set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Training data

# Validation set
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Validation data

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define AlexNet model
model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: malignant, benign, normal
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[ ]:


###VGG16 pre-trained model


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Image size for VGG16
img_width, img_height = 224, 224

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # VGG16 specific preprocessing
    rotation_range=20,              # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,          # Randomly shift images horizontally
    height_shift_range=0.2,         # Randomly shift images vertically
    shear_range=0.2,                # Shear transformation
    zoom_range=0.2,                 # Zoom in/out by 20%
    horizontal_flip=True,           # Randomly flip images horizontally
    validation_split=0.2)           # Split data into training and validation sets

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Only rescaling for test data

# Train set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Training data

# Validation set
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Validation data

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of VGG16
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: malignant, benign, normal
])

# Freeze the layers of the VGG16 base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[ ]:


###ResNet50


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# Image size for ResNet50
img_width, img_height = 224, 224

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet50 specific preprocessing
    rotation_range=20,              # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,          # Randomly shift images horizontally
    height_shift_range=0.2,         # Randomly shift images vertically
    shear_range=0.2,                # Shear transformation
    zoom_range=0.2,                 # Zoom in/out by 20%
    horizontal_flip=True,           # Randomly flip images horizontally
    validation_split=0.2)           # Split data into training and validation sets

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Only rescaling for test data

# Train set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Training data

# Validation set
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Validation data

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of ResNet50
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: malignant, benign, normal
])

# Freeze the layers of the ResNet50 base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[ ]:


##GoogLeNet Pre-trained model


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Image size for GoogLeNet (InceptionV3)
img_width, img_height = 299, 299

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # GoogLeNet specific preprocessing
    rotation_range=20,              # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,          # Randomly shift images horizontally
    height_shift_range=0.2,         # Randomly shift images vertically
    shear_range=0.2,                # Shear transformation
    zoom_range=0.2,                 # Zoom in/out by 20%
    horizontal_flip=True,           # Randomly flip images horizontally
    validation_split=0.2)           # Split data into training and validation sets

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Only rescaling for test data

# Train set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Training data

# Validation set
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Validation data

# Test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load pre-trained GoogLeNet (InceptionV3) model + higher level layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of GoogLeNet (InceptionV3)
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: malignant, benign, normal
])

# Freeze the layers of the GoogLeNet base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the labels of the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_labels = list(test_generator.class_indices.keys())

# Print Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=cm_labels))


# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

