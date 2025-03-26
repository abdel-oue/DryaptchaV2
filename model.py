import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import json
import pathlib

# Create the model function
def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 150, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(36 * 5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Process image for model prediction
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((150, 50))  # Resize to fit model input

    # Normalize the image
    image_data = np.array(image) / 255.0
    image_data = np.expand_dims(image_data, axis=-1)  # Add channel dimension

    return np.expand_dims(image_data, axis=0)  # Add batch dimension

# Encode label for the model
def encode_label(label):
    char_to_index = {chr(i): i - 97 for i in range(97, 123)}  # 'a' to 'z'
    char_to_index.update({str(i): i - 48 + 26 for i in range(10)})  # '0' to '9'

    label_tensor = [char_to_index[char] for char in label]
    return tf.one_hot(label_tensor, 36)  # One-hot encoding for each character

# Prepare training data
def prepare_data(images_dir, labels):
    image_paths = list(labels.keys())
    xs = []
    ys = []

    for image_name in image_paths:
        image_path = os.path.join(images_dir, image_name)
        label = labels[image_name]

        image_tensor = process_image(image_path)
        label_tensor = encode_label(label)

        xs.append(image_tensor)
        ys.append(label_tensor)

    return np.concatenate(xs), np.concatenate(ys)

# Training function
def train_model(images_dir, labels):
    model_path = './captcha_model/model.h5'
    
    # If the model already exists, skip training
    if os.path.exists(model_path):
        print('Model already exists. Skipping training...')
        return

    # Create a new model
    model = create_model()

    # Prepare the data
    xs, ys = prepare_data(images_dir, labels)

    # Train the model with the data
    model.fit(xs, ys, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(model_path)
    print('Model saved successfully.')

# Predict CAPTCHA
def predict_captcha(image_path):
    model_path = './captcha_model/model.h5'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print('Error: Model file not found! Train it first.')
        return

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Process the image to make it ready for prediction
    image = process_image(image_path)

    # Use the model to predict the captcha from the image
    prediction = model.predict(image)

    # Extract the predicted characters
    predicted_chars = []
    for i in range(5):
        start_idx = i * 36
        end_idx = start_idx + 36
        char_pred = prediction[0][start_idx:end_idx]
        predicted_index = np.argmax(char_pred)
        predicted_chars.append(decode_char(predicted_index))

    predicted_captcha = ''.join(predicted_chars)
    print(f'Predicted captcha: {predicted_captcha}')

# Decode index to character
def decode_char(index):
    if index < 26:
        return chr(index + 97)  # 'a' to 'z'
    elif index < 36:
        return chr(index - 26 + 48)  # '0' to '9'
    else:
        raise ValueError('Invalid index')

# Load labels from JSON file
def load_labels(labels_file):
    with open(labels_file, 'r') as file:
        return json.load(file)

# Main function to train and predict
def main():
    images_dir = './captchapngs'
    labels_file = './somecaptchas.json'
    labels = load_labels(labels_file)

    # Train the model
    train_model(images_dir, labels)

    # Predict a sample CAPTCHA
    predict_captcha('./captchapngs/captcha_1.png')

if __name__ == '__main__':
    main()
