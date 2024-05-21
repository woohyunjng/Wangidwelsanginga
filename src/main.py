import cv2

fase_cascade = cv2.CascadeClassifier("src/data/haarcascade_frontalface_default.xml")
if fase_cascade.empty():
    print("객체 생성 실패")
    exit()

RESIZE = (640, 640)

file_path = input("판별할 사진: ")
img = cv2.imread(f"src/image/test/{file_path}")
if img is None or img.size == 0:
    print("파일이 존재하지 않습니다")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
scale_factor = 1.3
faces = fase_cascade.detectMultiScale(gray, scale_factor, 5)

if not len(faces):
    print("얼굴이 감지되지 않았습니다")
    exit()

res = None

for x, y, w, h in faces:
    show_img = img.copy()
    cv2.rectangle(show_img, (x, y, w, h), (255, 0, 0), 5)

    if RESIZE is not None:
        show_img = cv2.resize(show_img, dsize=RESIZE, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image view", show_img)

    print("얼굴이 잘 감지됐나요? (Yes -> O / No -> 아무거나 클릭)")
    check = cv2.waitKey(0)

    if chr(check & 0xFF) != "O":
        continue
    res = (x, y, w, h)

if not res:
    print("얼굴이 감지되지 않았습니다")
    exit()

cropped = img[y : y + h, x : x + w]
if RESIZE is not None:
    resized_img = cv2.resize(cropped, dsize=RESIZE, interpolation=cv2.INTER_AREA)
else:
    resized_img = cropped.copy()
cv2.imshow("Image view", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
