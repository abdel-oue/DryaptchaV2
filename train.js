const fs = require('fs');
const path = require('path');
const readline = require('readline');
const tf = require('@tensorflow/tfjs-node');
const cliProgress = require('cli-progress');
const { createModel, prepareData } = require('./modelonly');

// Set up readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Set up progress bar
const progressBar = new cliProgress.SingleBar({
  format: 'Training | {bar} | {percentage}% | {value}/{total} Epochs',
  barCompleteChar: '\u2588',
  barIncompleteChar: '\u2591',
  hideCursor: true
}, cliProgress.Presets.shades_classic);

async function promptUserToReplaceModel() {
  return new Promise((resolve) => {
    rl.question('A model already exists. Do you want to replace it? (y/n): ', (answer) => {
      resolve(answer.toLowerCase() === 'y');
    });
  });
}

async function loadLabels(labelsPath) {
  try {
    const fullPath = path.resolve(labelsPath);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Labels file not found: ${fullPath}`);
    }
    const data = fs.readFileSync(fullPath, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    console.error('Error loading labels:', err);
    process.exit(1);
  }
}

async function trainOrLoadModel(imagesDir, labelsPath) {
    try {
      const labels = await loadLabels(labelsPath);
      
      console.log("Loaded Labels:", labels); // 🛠 Debugging Step
  
      if (!labels || Object.keys(labels).length === 0) {
        console.error('Error: Labels file is empty or invalid.');
        process.exit(1);
      }
  
      await trainModel(imagesDir, labels);
    } catch (err) {
      console.error('Error during training:', err);
    } finally {
      rl.close();
    }
  }
  
async function trainModel(imagesDir, labels) {
  const modelDir = path.join(__dirname, 'captcha_model'); // Model folder
  const modelPath = path.join(modelDir, 'model.json');

  // Check if the model already exists
  if (fs.existsSync(modelPath)) {
    const replaceModel = await promptUserToReplaceModel();
    if (!replaceModel) {
      console.log('Using existing model.');
      await loadModel(modelPath);
      return;
    }
  }

  console.log('Training a new model...');

  // Create a new model
  const model = createModel();
  const { xs, ys } = await prepareData(imagesDir, labels);

  // Ensure ys is float32
  const ysProcessed = ys.toFloat();

  // Training parameters
  const epochs = 10;
  progressBar.start(epochs, 0);
  console.log('xs shape:', xs.shape);
  console.log('ys shape:', ysProcessed.shape);

  for (let epoch = 0; epoch < epochs; epoch++) {
    await model.fit(xs, ysProcessed, {
      epochs: 1, // Train for 1 epoch at a time
      batchSize: 32,
      validationSplit: 0.2,
      verbose: 0
    });

    // Update progress bar
    progressBar.update(epoch + 1);
  }

  // Stop progress bar
  progressBar.stop();

  // Ensure model directory exists before saving
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }

  // Save the trained model
  await model.save(`file://${modelDir}`);
  console.log('Model trained and saved successfully.');
}

async function loadModel(modelPath) {
  try {
    const modelDir = path.dirname(modelPath);
    console.log(`Loading model from: file://${modelDir}`);
    const model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
    console.log('Model loaded successfully.');
  } catch (err) {
    console.error('Error loading model:', err);
  }
}

// Run the training or loading process
trainOrLoadModel('./captchapngs', './somecaptchas.json');