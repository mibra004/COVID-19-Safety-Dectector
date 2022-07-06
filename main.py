from dataclasses import field
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from multiprocessing import Queue
from keyboard import press
import numpy as np
import multiprocessing
import time
import cv2
import pyautogui
import argparse
import imutils
import os
def check(a, b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2      
    if 0 < dist < 0.5 * calibration:
        return True
    else:
        return False
def setup(yolo):
    global net, ln, LABELS
    weights = os.path.sep.join([yolo, "yolov3.weights"])
    config = os.path.sep.join([yolo, "yolov3.cfg"])
    labels_path = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")  
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def setup2(config, model, labels, size=416, confidence=0.5, threshold=0.3):
    global confidence1, threshold1, size1, labels1, net1
    confidence1 = confidence
    threshold1 = threshold
    size1 = size
    labels1 = labels
    net1 = cv2.dnn.readNetFromDarknet(config, model)
def inference_from_file(file):
    mat = cv2.imread(file)
    return inference(mat)
def inference(image):
    ih, iw = image.shape[:2]
    ln = net1.getLayerNames()
    ln = [ln[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (size1, size1), swapRB=True, crop=False)
    net1.setInput(blob)
    start = time.time()
    layer_outputs = net1.forward(ln)
    end = time.time()
    inference_time = end - start
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence1:
                box = detection[0:4] * np.array([iw, ih, iw, ih])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence1, threshold1)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            key = class_ids[i]
            confidence = confidences[i]
            results.append((key, labels1[key], confidence, x, y, w, h))
    return iw, ih, inference_time, results
def process(image):
    global processedImg, mask_violation, no_mask_violation, distance_violation, no_distance_violation, people, faces, closer
    mask_violation = 0
    no_mask_violation = 0
    distance_violation = 0
    no_distance_violation = 0
    people = 0
    faces = 0
    closer = 0
    (H, W) = (None, None)
    frame = image.copy()
    width, height, inference_time, results = inference(frame)
    for detection in results:
        key, name, confidence, x, y, w, h = detection
        if(key > 0):
            mask_violation += 1
        else:
            no_mask_violation +=1
        color = colors[key]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    confidences = []
    outline = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person" and confidence > 0.5: 
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                outline.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = []
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)
        for i in range(len(center)):
            for j in range(len(center)):
                close = check(center[i], center[j])
                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        people = len(box_line)
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 204), 2)
                closer += 1 
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 201, 45), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (22, 180, 231), 2)
    distance_violation = closer
    no_distance_violation = people - closer
    faces = mask_violation + no_mask_violation
    cv2.rectangle(frame,(0,0),(512,32),(0,0,0),-1)
    cv2.rectangle(frame,(1,1),(510,32),(255,255,255),2)
    combined_text1 = "Person(s) Detected        Social Distancing          Face Mask       Face(s) Detected"
    combined_text2 = "        {}          (Obeying: {} | Defying: {}) (Obeying: {} | Defying: {})        {}".format(people, no_distance_violation, distance_violation, no_mask_violation, mask_violation, faces)
    cv2.putText(frame, combined_text1, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 0)
    cv2.putText(frame, combined_text2, (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 0)
    field1 = people
    field2 = no_distance_violation
    field3 = distance_violation
    field4 = faces
    field5 = no_mask_violation
    field6 = mask_violation
    thingspeak = base_url + "&field1={}&field2={}&field3={}&field4={}&field5={}&field6={}".format(field1, field2, field3, field4, field5, field6)
    urlopen(thingspeak)
    processedImg = frame.copy()
def alert(queue):
    contact_name = "'Mustafa Atif Ibrahim'"
    search_for = "Mustafa Atif Ibrahim"
    message = queue.get()
    whatsapp_url = r'https://web.whatsapp.com/'
    opt = Options()
    opt.add_argument(r'user-data-dir=D:\Desktop Backup\COVID-19 Safety Detector\whatsapp-profile')
    driver = webdriver.Chrome(r'D:\Desktop Backup\COVID-19 Safety Detector\chromedriver\chromedriver.exe', options=opt)
    driver.get(whatsapp_url )
    time.sleep(30)
    search_box = driver.find_element(By.XPATH, "//div[@title='Search input textbox']")
    ActionChains(driver).move_to_element(search_box).click(search_box).perform()
    time.sleep(2)
    search_box.send_keys(search_for)
    time.sleep(2)
    chat_box = driver.find_element(By.XPATH, "//span[@title="+contact_name+"]")
    ActionChains(driver).move_to_element(chat_box).click(chat_box).perform()
    time.sleep(2)
    clip_button = driver.find_element(By.XPATH, "//span[@data-testid='clip']")
    ActionChains(driver).move_to_element(clip_button).click(clip_button).perform()
    time.sleep(2)
    image_button = driver.find_element(By.XPATH, "//span[@data-testid='attach-image']")
    ActionChains(driver).move_to_element(image_button).click(image_button).perform()
    time.sleep(2)
    pyautogui.typewrite(r"D:\Desktop Backup\COVID-19 Safety Detector\alert.jpg",0.1)
    time.sleep(2)
    press('enter')
    time.sleep(2)
    message_box = driver.find_element(By.XPATH, "//div[@data-testid='drawer-middle']//span//div//span//div//div//div//div//div//div//div//div//div//div//div[@role='textbox']")
    message_box.send_keys(message)
    time.sleep(2)  
    sending = driver.find_element(By.XPATH, "//span[@data-testid='send']")
    ActionChains(driver).move_to_element(sending).click(sending).perform()
    time.sleep(10)
    driver.close()
    driver.quit()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
    args = ap.parse_args()
    classes = ["Wearing Mask Properly", "Not Wearing Mask Properly", "Not Wearing Mask At All"]
    if args.network == "normal":
        setup2("models/mask-yolov4.cfg", "models/mask-yolov4.weights", classes)
    elif args.network == "prn":
        setup2("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)
    else:
        setup2("models/mask-yolov4-tiny.cfg", "models/mask-yolov4-tiny.weights", classes)
    global colors
    colors = [(55, 201, 45), (22, 180, 231), (50, 50, 204)]
    write_key = "MI46NVROLNG2XVXX"
    global base_url
    base_url = "https://api.thingspeak.com/update?api_key={}".format(write_key)
    create = None
    frameno = 0
    process_time = 0
    alert_image = "alert.jpg"
    input_filename = "input1.mp4"
    yolo = "yolo-coco/"
    output_filename = "output.avi"
    if(input_filename == ""):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_filename)
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=512, height=512)
        frameno += 1
        if(frameno%2 == 0 or frameno == 1):
            setup(yolo)
            process(current_img)
            processed_frame = processedImg
            cv2.imshow("COVID-19 Safety Violations Detector", processed_frame)
            if(time.time() - process_time >= 150 and process_time > 0):
                process1.kill()
            if(people > 0 and faces > 0 and time.time() - process_time >= 300):  
                if(float(distance_violation) / float(people) > 0.5 and float(mask_violation) / float(faces) > 0.5):
                    cv2.imwrite(alert_image, processed_frame)
                    queue1 = Queue()
                    process1 = multiprocessing.Process(target= alert, args=(queue1,))
                    process_time = time.time()
                    process1.start()
                    queue1.put('COVID-19 Safety Alert! Majority are defying face mask and social distancing rules. Please spread the COVID-19 safety rules of obeying face mask and social distancing.')
                else:
                    if(float(mask_violation) / float(faces) > 0.5):
                        cv2.imwrite(alert_image, processed_frame)
                        queue1 = Queue()
                        process1 = multiprocessing.Process(target= alert, args=(queue1,))
                        process_time = time.time()
                        process1.start()
                        queue1.put('COVID-19 Safety Alert! Majority are defying face mask rule. Please spread the COVID-19 safety rule of obeying face mask.')
                    elif(float(distance_violation) / float(people) > 0.5):
                        cv2.imwrite(alert_image, processed_frame)
                        queue1 = Queue()
                        process1 = multiprocessing.Process(target= alert, args=(queue1,))
                        process_time = time.time()
                        process1.start()
                        queue1.put('COVID-19 Safety Alert! Majority are defying social distancing rule. Please spread the COVID-19 safety rule of obeying social distancing.')
            if create is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                create = cv2.VideoWriter(output_filename, fourcc, 30, (processed_frame.shape[1], processed_frame.shape[0]), True)
        create.write(processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()