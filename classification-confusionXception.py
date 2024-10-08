import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Suppress specific warnings from Keras
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Set up directories
dataset_dir = './dataset'  # Replace with the actual path to your dataset

# Parameters
img_height, img_width = 299, 299  # Xception uses 299x299 images
batch_size = 32

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2)  # 80% for training, 20% for validation

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # Set as validation data

# Load the trained Xception model
model = load_model('trained_models/xception_lumpy_skin.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {test_acc * 100:.2f}%")

# Generate predictions on validation set
Y_pred = model.predict(validation_generator)
y_pred = [1 if y > 0.5 else 0 for y in Y_pred]

# Confusion matrix and classification report
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['Healthy', 'Lumpy']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
