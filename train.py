import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def train_or_load_model(images_dir, labels_path):
    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    if not labels:
        print("Labels file is empty or invalid.")
        return

    model_dir = './captcha_model'
    model_path = os.path.join(model_dir, 'model.keras')

    if os.path.exists(model_path):
        replace = input("A model already exists. Do you want to replace it? (y/n): ")
        if replace.lower() != 'y':
            print("Using existing model.")
            load_model(model_path)
            return

    train_model(images_dir, labels, model_dir)


def train_model(images_dir, labels, model_dir):
    model = create_model()
    xs, ys = prepare_data(images_dir, labels)
    ys = tf.cast(ys, tf.float32)

    epochs = 10
    print(f"xs shape: {xs.shape}")
    print(f"ys shape: {ys.shape}")

    for epoch in tqdm(range(epochs), desc="Training"):
        model.fit(xs, ys, epochs=1, batch_size=32, validation_split=0.2, verbose=0)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(os.path.join(model_dir, 'model.keras'))
    print("Model trained and saved successfully.")


def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)

train_or_load_model('./captchapngs', './somecaptchas.json')
