const Jimp = require('jimp');

async function processImage(imagePath) {
  try {
    // Load the image
    const image = await Jimp.read(imagePath);

    // Convert image to grayscale (no resizing)
    image.greyscale();

    // Save the processed image
    await image.writeAsync('grayscale-image.png');
    console.log('Image converted to grayscale and saved successfully!');
    
  } catch (err) {
    console.error('Error processing image:', err);
  }
}

processImage('C:/Users/pc/OneDrive/Desktop/Dryaptcha/captchapngs/captcha_1.png'); // Provide the correct path
