const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const fs = require('fs');
const path = require('path');

// Define your character set
const CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const CODE_LENGTH = 5; // Adjust based on the CAPTCHA length (in your case it's 5 characters)

// Load the CAPTCHA JSON file
function loadJsonData(filePath) {
  const data = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(data);
}

// Function to load and preprocess the image
async function loadImage(imagePath) {
  const image = await Jimp.read(imagePath);
  image.resize(150, 50).greyscale(); // Resize and convert to grayscale

  // Convert the image to a tensor
  const imageData = new Uint8Array(image.bitmap.width * image.bitmap.height);

  // Extract grayscale pixel values
  for (let i = 0; i < image.bitmap.width * image.bitmap.height; i++) {
    const pixel = image.bitmap.data[i * 4]; // Grayscale values are in the red channel
    imageData[i] = pixel;
  }

  // Return the image as a tensor with shape [1, 50, 150, 1]
  return tf.tensor(imageData, [1, 50, 150, 1], 'int32');
}

// Encode labels (e.g., "hxr87" -> [7, 23, 17, 8, 4])
function encodeLabel(label) {
  const encoded = [];
  for (let i = 0; i < label.length; i++) {
    const charIndex = CHARACTERS.indexOf(label[i]);
    encoded.push(charIndex);
  }
  return encoded;
}

// Define the CNN model for CAPTCHA extraction
function createModel() {
  const model = tf.sequential();

  // Conv layers
  model.add(tf.layers.conv2d({
    inputShape: [50, 150, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Flatten and dense layers
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // Output layer with softmax (one per character in code)
  model.add(tf.layers.dense({
    units: CHARACTERS.length * CODE_LENGTH,
    activation: 'softmax'
  }));

  return model;
}

// Train the model with the dataset
async function trainModel(jsonData) {
  const model = createModel();
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  const xs = [];
  const ys = [];

  // Load all images and labels
  for (const [imageName, label] of Object.entries(jsonData)) {
    const imagePath = path.join('./captchapngs', imageName); // Adjust the path to where your images are located
    const image = await loadImage(imagePath);
    const encodedLabel = encodeLabel(label);
    const oneHotLabel = tf.oneHot(encodedLabel, CHARACTERS.length);
    xs.push(image);
    ys.push(oneHotLabel);
  }

  const xsTensor = tf.stack(xs);
  const ysTensor = tf.stack(ys);

  // Train the model
  await model.fit(xsTensor, ysTensor, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2
  });

  // Save the model after training
  await model.save('file://captcha_model');
  console.log('Model saved!');
}

// Test the model on a new image
async function testModel(imagePath) {
  const model = await tf.loadLayersModel('labels.json');
  const image = await loadImage(imagePath);
  const prediction = model.predict(image);

  const predictedClass = prediction.argMax(1).dataSync();
  const predictedCode = predictedClass.join('');
  console.log(`Predicted CAPTCHA code: ${predictedCode}`);
}

// Example usage
const jsonData = loadJsonData('labels.json'); // Path to your JSON file

// Uncomment this to train the model
// trainModel(jsonData);

// Uncomment this to test the model
// testModel('./captchas/test_captcha.png');
