import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

dataset_dir = '../dataset'

img_height, img_width = 299, 299
batch_size = 32
epochs = 10

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
    subset='training')

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs)

model.save('xception_lumpy_skin.h5')

test_loss, test_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {test_acc * 100:.2f}%")

Y_pred = model.predict(validation_generator)
y_pred = [1 if y > 0.5 else 0 for y in Y_pred]

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['Healthy', 'Lumpy']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

def predict_image(image_path, model_path='xception_lumpy_skin.h5'):
    model = load_model(model_path)

    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction > 0.5:
        print(f"The cow has a lumpy skin disease with {prediction[0][0] * 100:.2f}% confidence.")
    else:
        print(f"The cow is healthy with {(1 - prediction[0][0]) * 100:.2f}% confidence.")

image_path = './1.jpg'
predict_image(image_path)
