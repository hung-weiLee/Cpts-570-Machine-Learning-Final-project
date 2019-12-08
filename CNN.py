from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

# initial CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dropout(0.2))
classifier.add(Dense(activation='sigmoid', units=1))

# Compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Preprocess
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

H = classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=50,
        validation_data=test_set,
        validation_steps=800)


plt.plot([n for n in range(1, len(H.history["accuracy"]) + 1)], H.history["accuracy"])
plt.title("CNN training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


plt.plot([n for n in range(1, len(H.history["accuracy"]) + 1)], H.history["val_accuracy"])
plt.title("CNN testing accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()