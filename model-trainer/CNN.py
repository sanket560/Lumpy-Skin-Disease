import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


dataset_dir = '../dataset'

# Parameters
img_height, img_width = 224, 224  # Custom CNN, Use 224x224 size
batch_size = 32
epochs = 10

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # Set as training data

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') 

# Build the custom CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs)

# Save the model
model.save('cnn_lumpy_skin.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {test_acc * 100:.2f}%")

# Predictions on validation set
Y_pred = model.predict(validation_generator)
y_pred = [1 if y > 0.5 else 0 for y in Y_pred]

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['Healthy', 'Lumpy']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

def predict_image(image_path, model_path='cnn_lumpy_skin.h5'):
    model = load_model(model_path)

    # Load the image
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    # Make the prediction
    prediction = model.predict(img_array)
    if prediction > 0.5:
        print(f"The cow has a lumpy skin disease with {prediction[0][0] * 100:.2f}% confidence.")
    else:
        print(f"The cow is healthy with {(1 - prediction[0][0]) * 100:.2f}% confidence.")

image_path = './1.jpg' 
predict_image(image_path)
