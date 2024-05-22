import cv2
import numpy as np

import json

with open("src/data/king.json", "r", encoding="utf8") as json_file:
    data_json = json.load(json_file)

fase_cascade = cv2.CascadeClassifier("src/data/haarcascade_frontalface_default.xml")
if fase_cascade.empty():
    print("객체 생성 실패")
    exit()

BASE_PATH = "src/image"
RESIZE = (720, 540)

scale_factor = 1.3
res = None

capture = cv2.VideoCapture(0)
while capture.isOpened():
    key = cv2.waitKey(1)
    if chr(key & 0xFF) == "q":
        exit()

    if chr(key & 0xFF) == "o":
        break

    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fase_cascade.detectMultiScale(gray, scale_factor, 5)

    if len(faces):
        x, y, w, h = faces[0]
        cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 5)
        res = (x, y, w, h)
    if RESIZE is not None:
        img = cv2.resize(img, dsize=RESIZE, interpolation=cv2.INTER_AREA)
    cv2.imshow("Camera", img)

if not res:
    print("얼굴이 감지되지 않았습니다")
    exit()

cropped = img
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
print("가장 비슷한 대통령의 얼굴을 찾는중...")

models = [cv2.face.LBPHFaceRecognizer.create() for i in range(len(data_json))]

ranks = []

for i in range(len(data_json)):
    length = data_json[i]["data"]
    faces, labels = [], []

    for j in range(length):
        img = cv2.imread(f"src/data/{i}/{j}.jpg", cv2.IMREAD_GRAYSCALE)
        faces.append(np.asarray(img, dtype=np.uint8))
        labels.append(j)

    models[i].train(np.asarray(faces), np.asarray(labels))
    res = models[i].predict(cropped)

    if res[1] < 500:
        confidence = 100 * (1 - res[1] / 300)
        ranks.append((confidence, i))

ranks = list(sorted(ranks, reverse=True))

king_data = ranks[0]
print(
    f"축하합니다! {data_json[king_data[1]]['name']}과 가장 유사한 얼굴을 갖고 있습니다 (유사도 {king_data[0]}%)"
)
print(
    ", ".join(
        [
            f"{i + 1}. {data_json[ranks[i][1]]['name']} (유사도 {ranks[i][0]}%)"
            for i in range(1, len(ranks))
        ]
    )
)
