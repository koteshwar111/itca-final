import cv2
import pytesseract
import re
import numpy as np
import os
import sys
import threading
from threading import Lock
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, request, jsonify


result_lock = Lock()
result_obtained = {"fine_amount": None, "stop_flag": False}
browser_instances = []

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

app = Flask(__name__)
is_running = False
receiver_email = ""
fine_amount = 0
log_messages = []
process_thread = None
log_lock = threading.Lock()

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def isValidVehicleNumberPlate(number_plate):
    regex = "^[A-Z]{2}[\\s-]?[0-9]{2}[\\s-]?[A-Z]{1,2}[\\s-]?[0-9]{4}$"
    return bool(re.match(regex, number_plate))

def formatVehicleNumber(vehicle_number):
    parts = re.findall(r'[A-Za-z]+|\d+', vehicle_number)
    return ' '.join(parts[:2]) + ' ' + ' '.join(parts[2:])


def extract_number_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return four_point_transform(gray, approx.reshape(4, 2))
    return None

def recognize_number_plate(number_plate_region):
    return pytesseract.image_to_string(number_plate_region, config='--psm 8').strip()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def send_email_notification(vehicle_number, owner_name, grand_total, to_email):
    try:
        grand_total = int(grand_total)
        fine_amount_int = int(fine_amount)  # Ensure fine_amount is an integer
    except ValueError as e:
        print(f"Error converting fine amount to integer: {e}")
        return

    if grand_total > fine_amount_int:
        from_email = "trafficchallanidentifier@gmail.com"
        subject = "Heavy Challan Vehicle"
        body = f"A Vehicle with Number: {vehicle_number} has a Challan of Rs. {grand_total}/-"

        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        password = "fzyh scbk pbkb bcal"

        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, message.as_string())
            log_message(f"Email sent to {receiver_email} successfully!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            server.quit()

log_history = set()

def log_message(message, unique_event=None):
    global log_messages, log_history
    with log_lock:
        if unique_event:
            if unique_event in log_history:
                return  
            log_history.add(unique_event)  
        log_messages.append(message)
        print(message)

def run_continuous_process(email, fine_amt):
    cap = cv2.VideoCapture(0)
    log_message("Heating up the Camera...")
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        number_plate_region = extract_number_plate(frame)
        if number_plate_region is not None:
            number_plate_text = recognize_number_plate(number_plate_region)
            if isValidVehicleNumberPlate(number_plate_text):
                fine_amount, owner_name = get_fine_amount(number_plate_text)
                if fine_amount == "No Pending Challans":
                    log_message("Restarting process for next vehicle.")
                    continue
                
                send_email_notification(number_plate_text, owner_name, fine_amount, email)

        cv2.imshow('Number Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def fetch_fine_amount_from_browser(vehicle_number, browser_id):
    global result_obtained, browser_instances

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--headless')  

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    browser_instances.append(driver)  # Track the browser instance

    try:
        driver.get("https://echallan.tspolice.gov.in/publicview/")
        owner_name_xpath = '/html/body/table[1]/tbody/tr[2]/td/div/div[1]/div/div/table/tbody/tr/td/div/form/table[1]/tbody/tr[1]/th[5]/div'

        vehicle_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "REG_NO")))
        vehicle_input.send_keys(vehicle_number)

        captcha_solved = False
        for _ in range(15):

            if result_obtained["stop_flag"]:
                print("Browser {browser_id} stopping as another browser found the result.")
                break
            for i in range(2, 10):
                try:
                    if result_obtained["stop_flag"]:
                        print("Browser {browser_id} stopping due to global stop flag.")
                        return "Browsing Stopped"

                    captcha_image = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#captchaDivtab1 img")))
                    captcha_src = captcha_image.get_attribute('src')

                    captcha_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "captchatab1")))
                    captcha_input.clear()
                    captcha_input.send_keys(str(i))

                    go_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "tab1btn")))
                    go_button.click()

                    try:
                        alert = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".sweet-alert.showSweetAlert.visible")))
                        alert_ok_button = alert.find_element(By.CSS_SELECTOR, ".confirm.btn.btn-lg.btn-primary")
                        alert_ok_button.click()
                        continue
                    except TimeoutException:
                        pass

                    try:
                        no_challans_message = driver.find_element(By.CSS_SELECTOR, "#tab1 div[style*='color: red; font-size: 16px;']")
                        if "No Pending Challans" in no_challans_message.text:
                            log_message(f"No Pending Challans on Vehicle {vehicle_number}.")
                            return "No Pending Challans"
                    except:
                        pass

                    owner_name_element = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, owner_name_xpath))
                    )
                    grand_total_element = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="rtable"]/tbody/tr[7]/td[4]/strong'))
                    )
                    owner_name = owner_name_element.text
                    
                    grand_total_text = grand_total_element.text if grand_total_element else "0"
                    fine_amount = int(grand_total_text)

                    with result_lock:
                        if result_obtained["fine_amount"] is None:
                            result_obtained["fine_amount"] = fine_amount
                            result_obtained["owner_name"] = owner_name
                            result_obtained["stop_flag"] = True
                            log_message(f"Vehicle Number: {vehicle_number}, Owner Name: {owner_name}, Fine Amount: {grand_total_text}")
                            return fine_amount

                except Exception:
                    continue

                if result_obtained["stop_flag"]:
                    break
            if result_obtained["stop_flag"]:
                break

    finally:
        driver.quit()
        browser_instances.remove(driver)  # Remove the browser instance from the list

    return "No Pending Challans"



# Cache to store vehicle fines: {vehicle_number: (fine_amount, owner_name)}
fine_cache = {}

result_obtained = {
    "fine_amount": None,
    "owner_name": None,
    "stop_flag": False  # To signal other browsers to stop when a result is found
}

def get_fine_amount(vehicle_number):
    # Format the vehicle number
    vehicle_number = formatVehicleNumber(vehicle_number)
    log_message(f"Detected Vehicle Number: {vehicle_number}", unique_event=vehicle_number)

    # Check if the fine for this vehicle number is already cached
    if vehicle_number in fine_cache:
        fine_amount, owner_name = fine_cache[vehicle_number]
        log_message(f"Detected Vehicle Number: {vehicle_number}")
        log_message(f"Vehicle Number: {vehicle_number}. Owner Name: {owner_name}, Fine Amount: {fine_amount}")
        return fine_cache[vehicle_number]  # Return cached fine amount and owner name

    num_browsers = 1
    futures = []
    result_obtained["fine_amount"] = None
    result_obtained["owner_name"] = None
    result_obtained["stop_flag"] = False
    browser_instances.clear()  # Clear existing browser instances

    # Use ThreadPoolExecutor to fetch fine amount using multiple browsers
    with ThreadPoolExecutor(max_workers=num_browsers) as executor:
        for i in range(num_browsers):
            futures.append(executor.submit(fetch_fine_amount_from_browser, vehicle_number, i))

        for future in as_completed(futures):
            # Break the loop early if another thread has already set the result
            if result_obtained["stop_flag"]:
                break
            
            result = future.result()

            # If a result is found, set it and stop other browsers
            if result_obtained["fine_amount"] is not None:
                result_obtained["stop_flag"] = True
                break

    # If fine amount is found, cache it before returning
    if result_obtained["fine_amount"] is not None:
        fine_cache[vehicle_number] = (result_obtained["fine_amount"], result_obtained["owner_name"])
        return result_obtained["fine_amount"], result_obtained["owner_name"]

    else:
        return "No Pending Challans"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global is_running, receiver_email, fine_amount, process_thread
    if not is_running:
        data = request.json
        receiver_email = data.get('email')
        fine_amount = int(data.get('fineAmount'))
        is_running = True
        log_message(f"Started with email: {receiver_email} and fine amount: {fine_amount}")
        process_thread = threading.Thread(target=run_continuous_process, args=(receiver_email, fine_amount))
        process_thread.start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already running"})

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    if is_running:
        is_running = False
        log_message("Process Stopped.")
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"status": "not running"})

@app.route('/logs', methods=['GET'])
def logs():
    return jsonify({"logs": log_messages})

if __name__ == '__main__':
    app.run(debug=True)
