const tf = require('@tensorflow/tfjs-node-gpu');

async function check() {
    const backend = tf.getBackend();
    console.log(`TensorFlow.js is using: ${backend}`);
}

check();
