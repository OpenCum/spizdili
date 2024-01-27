import cv2
import imageio
from imutils.video import VideoStream


vs = VideoStream(src=0).start()

color = (0, 255, 0)  # Зелений колір


face_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\new\\haarcascade_frontalface_default.xml')


video_writer = imageio.get_writer('C:\\Users\\user\\facevideo.mp4', fps=60)

timer = 0


window_name = "Frame"

# Циклічно отримуємо кадри з камери
while True:
    # Отримуємо кадр з камери
    frame = vs.read()

    # Перетворюємо кадр у відтінки сірого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Застосовуємо детектор обличчя
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Малюємо зелений прямокутник навколо обличчя
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Додаємо кадр до відео
            video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Скидаємо таймер
        timer = 30
    else:
        # Зменшуємо таймер, якщо немає обличчя
        timer = max(0, timer - 1)

    # Відображаємо кадр, тільки якщо таймер не завершився
    if timer > 0:
        cv2.imshow(window_name, frame)
    else:
        # Перевіряємо, чи існує вікно перед тим, як його знищити
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(window_name)

    # Якщо натиснуто клавішу ESC, виходимо з програми
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


vs.stop()
video_writer.close()


cv2.destroyAllWindows()
