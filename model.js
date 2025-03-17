const tf = require('@tensorflow/tfjs-node-gpu');
const Jimp = require('jimp');
const path = require('path');

// Create model function
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [50, 150, 1],  // Image size
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 36, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// Load and preprocess image
async function processImage(imagePath) {
  const image = await Jimp.read(imagePath);
  image.resize(150, 50).greyscale();

  const imageData = new Uint8Array(image.bitmap.width * image.bitmap.height);
  for (let i = 0; i < image.bitmap.width * image.bitmap.height; i++) {
    const pixel = image.bitmap.data[i * 4];
    imageData[i] = pixel;
  }

  return tf.tensor(imageData, [1, 50, 150, 1], 'int32');
}

// Prepare data
async function prepareData(imagesDir, labels) {
  const imagePaths = Object.keys(labels);
  const xs = [];
  const ys = [];

  for (let i = 0; i < imagePaths.length; i++) {
    const imagePath = path.join(imagesDir, imagePaths[i]);
    const label = labels[imagePaths[i]];

    const imageTensor = await processImage(imagePath);
    const labelTensor = tf.oneHot(tf.tensor1d([label], 'int32'), 36);  // One-hot encoding for 36 classes

    xs.push(imageTensor);
    ys.push(labelTensor);
  }

  return {
    xs: tf.concat(xs),
    ys: tf.concat(ys),
  };
}

// Training function
async function trainModel(imagesDir, labels) {
  const model = createModel();
  const { xs, ys } = await prepareData(imagesDir, labels);

  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 4,
    validationSplit: 0.2,
  });

  await model.save('file://./captcha_model/model.json');
}

// Prediction function
async function predictCaptcha(imagePath) {
  const model = await tf.loadLayersModel('file://./captcha_model/model.json');
  const image = await processImage(imagePath);
  const prediction = model.predict(image);
  const predictedClass = prediction.argMax(1).dataSync()[0];
  console.log(`Predicted captcha: ${predictedClass}`);
}

// Example usage
const imagesDir = './captchapngs';
const labels = require('./somecaptchas.json');  // Load labels from the labels.json file

trainModel(imagesDir, labels)
  .then(() => {
    console.log('Model trained and saved.');
    predictCaptcha('./captchapngs/captcha_1.png');  // Test with a sample captcha
  })
  .catch(err => {
    console.error('Error during training or prediction:', err);
  });