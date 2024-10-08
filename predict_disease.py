import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress warnings and logs
tf.get_logger().setLevel('ERROR')

image_path = 'sample.jpg'
model_dir = './trained_models'
logs_dir = './Logs'

# Check model folder exist
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory {model_dir} does not exist")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Model file names
model_files = {
    "DenseNet": ('densenet_lumpy_skin.h5', (224, 224)),
    "Xception": ('xception_lumpy_skin.h5', (299, 299)),
    "VGG19": ('vgg19_lumpy_skin.h5', (224, 224)),
    "CNN": ('cnn_lumpy_skin.h5', (224, 224)),
    "ResNet50": ('resnet50_lumpy_skin.h5', (224, 224))
}

# Load models
models = {}
for model_name, (model_file, input_size) in model_files.items():
    model_path = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    models[model_name] = (load_model(model_path), input_size)


# Use each model for prediction
def predict_with_models(image_path, models):
    predictions = {}
    for model_name, (model, input_size) in models.items():
        # Load and preprocess the image
        img = load_img(image_path, target_size=input_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the image

        # Make prediction
        prediction = model.predict(img_array)
        predictions[model_name] = prediction[0][0] * 100

    return predictions


# Get predictions
predictions = predict_with_models(image_path, models)

# Display predictions
for model_name, prediction in predictions.items():
    if prediction > 50:
        print(f"{model_name} Prediction: Lumpy Skin Disease: {prediction:.2f}%")
    else:
        print(f"{model_name} Prediction: Healthy: {100 - prediction:.2f}%")

# Create bar graph
model_names = list(predictions.keys())
prediction_values = list(predictions.values())

plt.figure(figsize=(12, 6))
colors = plt.get_cmap('tab10').colors
plt.bar(model_names, prediction_values, color=colors[:len(model_names)])
plt.xlabel('Model Name')
plt.ylabel('Prediction Percentage')
plt.title('Model Predictions for Lumpy Skin Disease')
plt.ylim([0, 110])

plt.yticks(np.arange(0, 101, 10))

for i, txt in enumerate(prediction_values):
    plt.text(model_names[i], prediction_values[i] + 1, f'{txt:.2f}%', ha='center', va='bottom')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
png_path = os.path.join(logs_dir, f'log_{timestamp}.png')
csv_path = os.path.join(logs_dir, f'log_{timestamp}.csv')

plt.savefig(png_path)
plt.close()

df = pd.DataFrame(list(predictions.items()), columns=['Model Name', 'Prediction Percentage'])
df.to_csv(csv_path, index=False)
print(f"Bar graph saved to {png_path}")
print(f"CSV file saved to {csv_path}")
