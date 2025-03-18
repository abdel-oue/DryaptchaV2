const tf = require('@tensorflow/tfjs-node-gpu');
const Jimp = require('jimp');
const path = require('path');
const fs = require('fs');

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
async function prepareData(imagesDir, labels) {
  const imagePaths = Object.keys(labels);
  const xs = [];
  const ys = [];

  for (const imageName of imagePaths) {
    const imagePath = path.join(imagesDir, imageName);
    const label = labels[imageName];

    const imageTensor = await processImage(imagePath);

    const labelTensor = encodeLabel(label);

    xs.push(imageTensor);
    ys.push(labelTensor);
  }
  return {
    xs: tf.concat(xs),
    ys: tf.stack(ys), 
  };
}
// Training function
async function trainModel(imagesDir, labels) {
  const modelPath = path.join(__dirname, 'captcha_model', 'model.json');
  // If the model already exists, skip training
  if (fs.existsSync(modelPath)) {
    console.log('Model already exists. Skipping training...');
    return;
  }

  // Create a new model
  const model = createModel();
  const { xs, ys } = await prepareData(imagesDir, labels);

  // Train the model with the data
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32, 
    validationSplit: 0.2,
  });

  // Save the trained model to disk
  await model.save(`file://${modelPath}`);
  console.log('Model saved successfully.');
}
async function predictCaptcha(imagePath) {
  const modelPath = path.join(__dirname, 'captcha_model', 'model.json');
  
  // Check if model exists
  if (!fs.existsSync(modelPath)) {
    console.error('Error: Model file not found! Train it first.');
    return;
  }

  try {
    // Load the trained model
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    // Process the image to make it ready for prediction
    const image = await processImage(imagePath);

    // Use the model to predict the captcha from the image
    const prediction = model.predict(image);

    // Extract the predicted characters
    const predictedChars = [];
    const predData = prediction.dataSync();
    for (let i = 0; i < 5; i++) {
      const startIdx = i * 36;
      const endIdx = startIdx + 36;
      const charPred = predData.slice(startIdx, endIdx);
      const predictedIndex = tf.argMax(tf.tensor1d(charPred), 0).dataSync()[0];
      predictedChars.push(decodeChar(predictedIndex));
    }

    const predictedCaptcha = predictedChars.join('');
    console.log(`Predicted captcha: ${predictedCaptcha}`);
  } catch (err) {
    console.error('Error loading model:', err);
  }
}
function decodeChar(index) {
  if (index < 26) {
    return String.fromCharCode(index + 97); // 'a' -> 0, 'z' -> 25
  } else if (index < 36) {
    return String.fromCharCode(index - 26 + 48); // '0' -> 26, '9' -> 35
  }
  throw new Error('Invalid index');
}
const imagesDir = './captchapngs';
const labels = require('./somecaptchas.json');

trainModel(imagesDir, labels)
  .then(() => {
    console.log('Model trained and saved.');
    return predictCaptcha('./captchapngs/captcha_1.png');
  })
  .catch(err => {
    console.error('Error during training or prediction:', err);
  });
