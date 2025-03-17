const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

// Create a folder to store the CAPTCHAs
const captchaFolder = path.join(__dirname, 'captchas');
if (!fs.existsSync(captchaFolder)) {
    fs.mkdirSync(captchaFolder);
}

// The JSON file to store the answers
const jsonFile = path.join(__dirname, 'captchas.json');

// Initialize JSON file if not already created
if (!fs.existsSync(jsonFile)) {
    fs.writeFileSync(jsonFile, JSON.stringify({}, null, 2));
}

async function collectCaptchaImagesAndSolve() {
    const browser = await puppeteer.launch({ 
        headless: false, 
        args: [
            '--start-maximized', 
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ],
    });

    const page = await browser.newPage();

    // Set viewport to Full HD resolution (adjust as needed for your screen size)
    await page.setViewport({ width: 1920, height: 1080 }); 

    // Go to the login page
    await page.goto('https://enr.tax.gov.ma/enregistrement/login', { waitUntil: 'networkidle2' });

    // Enter username and password
    const USERNAME1 = 'your_username_here';  // Replace with actual username
    const PASSWORD = 'your_password_here';  // Replace with actual password
    await page.type('#j_username', USERNAME1);
    await page.type('#j_password', PASSWORD);

    // Create an object to store the answers
    let captchaAnswers = {};

    // Loop to collect and solve multiple CAPTCHAs
    for (let i = 73; i <= 500; i++) {
        const captchaId = `captcha${i}`;

        // Find CAPTCHA image
        const captchaElement = await page.$('img[src="/enregistrement/captcha"]');
        if (!captchaElement) {
            console.error(`❌ CAPTCHA image for ${captchaId} not found!`);
            break;
        }

        // Take a screenshot of the CAPTCHA
        const captchaImagePath = path.join(captchaFolder, `${captchaId}.png`);
        await captchaElement.screenshot({ path: captchaImagePath });

        // Use Tesseract.js to solve the CAPTCHA
        const { data: { text } } = await Tesseract.recognize(
            captchaImagePath,
            'eng',  // You can change the language if needed
            {
                logger: (m) => console.log(m),
            }
        );

        // Log the text result
        console.log(`Tesseract recognized text: ${text}`);

        // Check if Tesseract returns valid text
        if (!text || text.trim() === '') {
            console.error(`❌ Invalid CAPTCHA text for ${captchaId}`);
            continue; // Skip this CAPTCHA if no valid result
        }

        // Store the answer in the JSON object
        captchaAnswers[captchaId] = text.trim().toUpperCase();

        console.log(`Solved ${captchaId}: ${captchaAnswers[captchaId]}`);

        // Go back to login page (refresh or navigate back to get new CAPTCHA)
        await page.reload({ waitUntil: 'networkidle2' });
    }

    // Save the answers to the JSON file
    fs.writeFileSync(jsonFile, JSON.stringify(captchaAnswers, null, 2));

    await browser.close();
}

collectCaptchaImagesAndSolve()
    .then(() => console.log('Process completed successfully!'))
    .catch((error) => console.error('Error:', error));