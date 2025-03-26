import tensorflow as tf
import numpy as np
import cv2
import os

class IndexMapper:
    def __init__(self):
        self.class_index_mapping = {}
        self.next_class_index = 0

    def get_class_index(self, label):
        if label not in self.class_index_mapping:
            self.class_index_mapping[label] = self.next_class_index
            self.next_class_index += 1
        return self.class_index_mapping[label]

    def num_classes(self):
        return len(self.class_index_mapping)

# Create model function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 150, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(36 * 5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (150, 50))
    image = image.astype(np.float32) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return np.expand_dims(image, axis=0)  # Add batch dimension

def encode_label(label):
    char_to_index = lambda char: ord(char) - ord('a') if 'a' <= char <= 'z' else ord(char) - ord('0') + 26
    label_tensor = [char_to_index(char) for char in label]
    return tf.one_hot(label_tensor, 36)

def prepare_data(images_dir, labels):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    images = []
    label_indices = []

    index_mapper = IndexMapper()

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (150, 50))
        image = image.astype(np.float32) / 255.0  # Normalize image

        images.append(image)

        # Get the label from the filename
        label = labels.get(image_file, '')

        label_index = index_mapper.get_class_index(label)
        label_indices.append(label_index)

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    label_indices = np.array(label_indices)
    ys = tf.one_hot(label_indices, index_mapper.num_classes())

    print(f"xs shape: {images.shape}")
    print(f"ys shape: {ys.shape}")

    return images, ys

# Example of using the functions
if __name__ == "__main__":
    labels = {
        'captcha1.png': 'ne3rr',
        'captcha2.png': '2gjqk',
        # Add other label mappings here
    }

    images_dir = 'path_to_images'  # Specify the path to your images
    images, ys = prepare_data(images_dir, labels)

    model = create_model()
    model.fit(images, ys, epochs=10, batch_size=32)
