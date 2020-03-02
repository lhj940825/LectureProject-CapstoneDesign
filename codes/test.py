import numpy as np
import cv2 as cv

'데이터 갯수 13233개'
xs = np.load('face_data/matrix.npy')    # 이미지 matrix (152, 152, 3)
ys = np.load('face_data/one_hot.npy')   # one hot 라벨 (5749, 1)

'이 둘은 의미없는 데이터입니다(디버깅용)'
names = np.load('face_data/name.npy')   # 이미지 라벨 사람이들
idx   = np.load('face_data/index.npy')  # 이미지 라벨 사람이름의 숫자

print("MATRIX", xs[777])
print("ONE_HOT", ys[777])
print("NAME", names[777])
print("INDEX", idx[777])

cv.imshow("%s (%s)" % (names[1], idx[1]), xs[1])
cv.waitKey(0)

