import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from model import DTN
from PIL import Image
# 라이브러리 등록



font = cv2.FONT_HERSHEY_SIMPLEX


# 폰트 등록



def faceDetect():
    face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')

    # 영상에서 인식할 소스 등록
    if face_cascade.empty(): raise Exception("your cascade is empty. are you sure, the path is correct ?")


    try:

        cap = cv2.VideoCapture(0)

    # 웹캠 활성화시키는 코드



    except:

        print('카메라 로딩 실패')

        return


    model = DTN(mode='eval', learning_rate=0.001)


    # build model
    model
    model.build_model(1)


    with tf.Session(config=tf.ConfigProto()) as sess:

        saver = tf.train.Saver()
        saver.restore(sess,'image-change/0611/dtn-42000')

        while True:

            ret, frame = cap.read()

            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 2, 0, (30, 30))

            for (x, y, w, h) in faces:
                shape = np.shape(frame[y:y+h, x:x+w])
                batch_images = frame[y:y+h, x:x+w]
                batch_images = cv2.resize(batch_images, (152, 152))
                batch_images=[batch_images/127.5 -1]

                print(shape)
                feed_dict = {model.images: batch_images}





                sampled_batch_images = ((sess.run(model.sampled_images, feed_dict))+1)*127.5

                sampled_batch_images = cv2.resize(sampled_batch_images[0], (h,w))
                print(sampled_batch_images)
                frame[y:y+h, x:x+w] = sampled_batch_images

                #
                # if type(faces) == np.ndarray:
                #     print(frame[y-10:y+h+10, x-10:x+w+10])
                #
                #     print('shape is'+ str(np.shape(frame[y-10:y+h+10, x-10:x+w+10])))
                #
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3, 4, 0)
                #
                # cv2.putText(frame, 'Detected Face', (x - 5, y - 5), font, 0.9, (255, 255, 0), 2)

            # 얼굴을 인식하는 사각형에 대한 소스, 텍스트 소스





            cv2.imshow('frame', frame)

            # 영상을 출력하는 소스


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

        cap.release()

        cv2.destroyAllWindows()


# space를 누르면 실행 종료되는 코드

# try:
faceDetect()
# except Exception as inst:
#     print("~~")
#     print(inst)


#face_cascade.load('./data/haarcascades/haarcascade_frontalface_default.xml')