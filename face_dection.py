import cv2
import numpy as np

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import face_recognition


def aHash(img):
    dim = (8, 8)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            s = s+gray[i, j]
    avg = s/64
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


def dHash(img):
    dim = (9, 8)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


def pHash(img):
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def orb_cal(img0, img1):
    # dim = (32, 32)
    # img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
    # img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    # print(img0.shape,img1.shape)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray0, None)
    kp2, des2 = orb.detectAndCompute(gray1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    similary = len(good) / len(matches)
    return similary


def norm_cal(img0, img1):
    dim = (32, 32)
    img0 = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    n0 = np.array(img0)
    n1 = np.array(img1)
    return np.linalg.norm(n0-n1)


def run(img0, img1):
    hash0 = aHash(img0)
    hash1 = aHash(img1)
    n1 = cmpHash(hash0, hash1)
    print('均值哈希算法相似度aHash：', n1)

    hash0 = dHash(img0)
    hash1 = dHash(img1)
    n2 = cmpHash(hash0, hash1)
    print('差值哈希算法相似度dHash：', n2)

    hash0 = pHash(img0)
    hash1 = pHash(img1)
    n3 = cmpHash(hash0, hash1)
    print('感知哈希算法相似度pHash：', n3)

    n4 = classify_hist_with_split(img0, img1)
    print('三直方图算法相似度：', n4)

    n5 = calculate(img0, img1)
    print("单通道的直方图", n5)

    # n6 = orb_cal(img0, img1)
    # print('ORB的相似度:', n6)

    n7 = norm_cal(img0, img1)
    print('欧氏距离:', n7)

    print("%.2f %.2f %.2f %.2f %.2f" % (1-float(n1/64), 1 -
                                             float(n2/64), 1-float(n3/64),
                                             n4, n5))

    plt.subplot(121)
    plt.imshow(Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)))
    plt.subplot(122)
    plt.imshow(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    plt.show()


def face_reg(path0, path1):
    known_image = face_recognition.load_image_file(path0)
    unknown_image = face_recognition.load_image_file(path1)
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces(
        [biden_encoding], unknown_encoding, tolerance=0.50)
    print(results)


if __name__ == "__main__":
    path0 = 'face/img11.jpg'
    path1 = 'face/img8.jpg'
    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)
    run(img0, img1)
    face_reg(path0, path1)