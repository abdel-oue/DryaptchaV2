import os
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
import pytesseract

# Set path for Tesseract-OCR (Update this path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create a folder to store CAPTCHAs
captcha_folder = os.path.join(os.getcwd(), "captchas")
os.makedirs(captcha_folder, exist_ok=True)

# JSON file to store answers
json_file = os.path.join(os.getcwd(), "captchas.json")

# Load existing data if JSON file exists
if os.path.exists(json_file):
    with open(json_file, "r") as file:
        captcha_answers = json.load(file)
else:
    captcha_answers = {}

# Set up Chrome WebDriver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-setuid-sandbox")

# Provide the path to ChromeDriver
service = Service("chromedriver.exe")  # Update path if necessary
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open your website
driver.get("https://enr.tax.gov.ma/enregistrement/login")
time.sleep(3)

# Enter credentials
USERNAME = "your_username_here"
PASSWORD = "your_password_here"
driver.find_element(By.ID, "j_username").send_keys(USERNAME)
driver.find_element(By.ID, "j_password").send_keys(PASSWORD)

# Loop to collect and solve multiple CAPTCHAs
for i in range(73, 500):
    captcha_id = f"captcha{i}"

    try:
        # Find CAPTCHA element
        captcha_element = driver.find_element(By.XPATH, '//img[@src="/enregistrement/captcha"]')
        captcha_image_path = os.path.join(captcha_folder, f"{captcha_id}.png")

        # Take a screenshot
        captcha_element.screenshot(captcha_image_path)

        # Solve CAPTCHA with Tesseract
        captcha_text = pytesseract.image_to_string(Image.open(captcha_image_path), config="--psm 6").strip().upper()

        if not captcha_text:
            print(f"❌ Invalid CAPTCHA text for {captcha_id}")
            continue

        # Store the answer
        captcha_answers[captcha_id] = captcha_text
        print(f"Solved {captcha_id}: {captcha_text}")

        # Save updated JSON file
        with open(json_file, "w") as file:
            json.dump(captcha_answers, file, indent=2)

        # Refresh page to get new CAPTCHA
        driver.refresh()
        time.sleep(3)

    except Exception as e:
        print(f"Error processing {captcha_id}: {e}")
        break

# Close browser
driver.quit()
print("Process completed successfully!")
