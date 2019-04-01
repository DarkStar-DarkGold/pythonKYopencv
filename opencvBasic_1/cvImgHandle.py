import time

import cv2

import numpy as np

img = cv2.imread('hhh.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)
print(img[10, 10])

print(img.item(10, 10, 1))
img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))

print(img.shape)  # 行数列数通道数
print(img.size)  # 像素个数\
print(img.dtype)

face = img[150:200, 0:100]  # 这里每两个值分别是x之间的距离  和  y之间的距离
print('/////', face)
# img[50:100,100:200] = face
# cv2.imshow('a',img)  # 输出看一下效果！
# cv2.waitKey(0)

# b = 0
# img[:,:,2] = b
# img[:,:,0] = b
# cv2.imshow('a',img)  # 输出看一下效果！
# cv2.waitKey(0)


from matplotlib import pyplot as plt

BLUE = [255, 0, 0]
replicate = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
# plt.show()


# img1 = cv2.imread('hhh.jpg')
# img2 = cv2.imread('timg.jpeg')
#
# dst = cv2.addWeighted(img1,0.7,img2,0.3,40)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 加载图片   不是很明白  按位于
# img1  = cv2.imread('hhh.jpg')
# img2 = cv2.imread('opencvLog.jpg')
#
# # 获取与logo的大小
# rows, cols, channels = img2.shape
# # 在大图上边选定roi区域
# roi = img1[0:rows,0:cols]
#
# # 创建
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# # 二值化
# ret ,mask = cv2.threshold(img2gray, 175 , 255, cv2.THRESH_BINARY )
#
# cv2.imshow('mask',mask)
# mask_inv = cv2.bitwise_not(mask) # 求逆
# cv2.imshow('mask1',mask_inv)
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
# # cv2.imwrite('./bg1_add.jpg',img1_bg)
# # cv2.imshow('bg1',img1_bg)
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
# cv2.imshow('bg2',img2_fg)
#
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
#
# cv2.imshow('res',img1)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img1 = cv2.imread('bg1.jpg')
e1 = cv2.getTickCount()
a = [1, 3, 5, 7, 9, 11]
# for i in range(5,49,2):
for i in a:
    img1 = cv2.medianBlur(img1, i)
    str1 = "s" + str(i)
    print(str1)
    cv2.imshow(str1, img1)
e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print(t)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(dir(cv2))

# 在 OpenCV 的 HSV 格式中，H(色彩/色度)的取值范围是 [0，179]， S(饱和度)的取值范围 [0，255]，
# V(亮度)的取值范围 [0，255]。但是不 同的软件使用的值可能不同。
# 所以当你需要拿 OpenCV 的 HSV 值与其他软 件的 HSV 值进行对比时，一定要记得归一化。
