// model.js
const tf = require('@tensorflow/tfjs-node-gpu');
const Jimp = require('jimp');
const path = require('path');
const fs = require('fs');
const cv = require('opencv4nodejs');

let classIndexMapping = {};
let nextClassIndex = 0;

const numClasses = () => Object.keys(classIndexMapping).length; // Get the number of classes dynamically

// Create model function
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [50, 150, 1], // Image size
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.dense({ units: 36 * 5, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function processImage(imagePath) {
  const image = await Jimp.read(imagePath);
  image.resize(150, 50).greyscale();

  const imageData = new Float32Array(image.bitmap.width * image.bitmap.height);
  for (let y = 0; y < image.bitmap.height; y++) {
    for (let x = 0; x < image.bitmap.width; x++) {
      const idx = (y * image.bitmap.width + x) * 4; // RGBA
      imageData[y * image.bitmap.width + x] = image.bitmap.data[idx] / 255.0; // Normalize
    }
  }

  return tf.tensor4d(imageData, [1, 50, 150, 1], 'float32');
}

function encodeLabel(label) {
  const charToIndex = (char) => {
    if (char >= 'a' && char <= 'z') {
      return char.charCodeAt(0) - 97; // 'a' -> 0, 'z' -> 25
    } else if (char >= '0' && char <= '9') {
      return char.charCodeAt(0) - 48 + 26; // '0' -> 26, '9' -> 35
    }
    throw new Error('Invalid character in CAPTCHA solution');
  };

  const labelTensor = label.split('').map(charToIndex);

  return tf.oneHot(tf.tensor1d(labelTensor, 'int32'), 36).reshape([5 * 36]);
}

// Prepare data: load images and labels, process them
async function prepareData(imagesDir, labels) {
    const imageFiles = fs.readdirSync(imagesDir);
    const imagePaths = imageFiles.filter(file => file.endsWith('.png'));
    
    const images = [];
    const labelIndices = []; // To store the integer class indices
  
    imagePaths.forEach(imageFile => {
      const imagePath = path.join(imagesDir, imageFile);
      const image = cv.imread(imagePath); // Load the image using OpenCV
      const resizedImage = cv.resize(image, new cv.Size(150, 50)); // Resize image to 150x50
      const normalizedImage = resizedImage.getDataAsArray(); // Normalize pixel values if necessary
  
      images.push(normalizedImage);
  
      // Get the label from the filename
      const label = labels[imageFile]; // e.g., 'ne3rr' for 'captcha4.png'
  
      // Check if label already exists in mapping, otherwise add it dynamically
      if (!(label in classIndexMapping)) {
        classIndexMapping[label] = nextClassIndex++;
      }
      
      // Push the index corresponding to the label
      labelIndices.push(classIndexMapping[label]);
    });
  
    // Convert the images array to a tensor (shape: [batch_size, height, width, channels])
    const xs = tf.tensor4d(images); 
  
    // Convert the label indices to a one-hot encoded tensor
    const ys = tf.oneHot(tf.tensor1d(labelIndices, 'int32'), numClasses()).toFloat(); // One-hot encode the labels
  
    console.log('xs shape:', xs.shape); // Should be [batch_size, height, width, channels]
    console.log('ys shape:', ys.shape); // Should be [batch_size, numClasses]
  
    return { xs, ys };
}
   
module.exports = {
  createModel,
  prepareData,
};
