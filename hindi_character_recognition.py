# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Extract the dataset
from zipfile import ZipFile
import os

file_name = '/content/drive/MyDrive/Project/Hindi_Character_Recognition.zip'
extract_path = '/content/Hindi_Character_Recognition'

if not os.path.exists(extract_path):
    with ZipFile(file_name, 'r') as zip:
        zip.extractall('/content/')
        print('Dataset extracted successfully.')
else:
    print('Dataset already extracted.')

# Check if the extraction path exists
if not os.path.exists(extract_path):
    print("Error: Dataset extraction failed. Please check the zip structure.")
else:
    print("Dataset found!")

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Use tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, accuracy_score


# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

training_set = train_datagen.flow_from_directory(
    '/content/Hindi_Character_Recognition/Training Data',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    '/content/Hindi_Character_Recognition/Testing Data',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(training_set.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    training_set,
    epochs=30,
    steps_per_epoch=len(training_set),
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Evaluate model on training and test data
train_loss, train_accuracy = model.evaluate(training_set)
test_loss, test_accuracy = model.evaluate(test_set)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Generate confusion matrix
y_pred = np.argmax(model.predict(test_set), axis=-1)
true_classes = test_set.classes

cm = confusion_matrix(true_classes, y_pred)
accuracy = accuracy_score(true_classes, y_pred)

print(f'Test Accuracy (via confusion matrix): {accuracy * 100:.2f}%')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
class_names = list(test_set.class_indices.keys())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training and validation accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Random image prediction
from keras.preprocessing import image
import random

def load_random_image(directory, target_size=(64, 64)):
    classes = [cls for cls in os.listdir(directory) if os.path.isdir(os.path.join(directory, cls))]
    random_class = random.choice(classes)
    class_path = os.path.join(directory, random_class)
    valid_images = [img for img in os.listdir(class_path) if os.path.splitext(img)[1].lower() in ['.png', '.jpg', '.jpeg']]
    if not valid_images:
        print(f"No valid images in class {random_class}.")
        return None, None, None
    image_name = random.choice(valid_images)
    image_path = os.path.join(class_path, image_name)

    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img, img_array, random_class

# Test with a random image
img, img_array, true_class = load_random_image('/content/Hindi_Character_Recognition/Testing Data')
if img is not None:
    predicted_class = np.argmax(model.predict(img_array), axis=-1)
    predicted_class_name = {v: k for k, v in test_set.class_indices.items()}[predicted_class[0]]

    plt.imshow(img)
    plt.title(f'True class: {true_class}\nPredicted class: {predicted_class_name}')
    plt.axis('off')
    plt.show()
