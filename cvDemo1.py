
import cv2
from matplotlib import pyplot as plt

# 图片的显示和读取和保存退出
'''
cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
img = cv2.imread("hhh.jpg",1)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('huangxiaoming.png', img)
    cv2.destroyAllWindows()'''

#  关于matplotlib显示图片  和 opencv读取彩色图像的bgr  和 plt显示图片的rgb的转换问题
'''
img = cv2.imread('hhh.jpg',1)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img2,interpolation = 'nearest')
plt.xticks([]), plt.yticks([])
plt.show()
'''
#  保存视频
'''
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 30
size = (int(cap.get(3)),
        int(cap.get(4)))
# fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
out = cv2.VideoWriter('output.mp4',fourcc, fps, size )
#
while True:
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        a = out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()'''

#  import numpy as np
# 2 import cv2
# 3
# 4 # Create a black image
# 5 img=np.zeros((512,512,3), np.uint8)
# 6
# 7 # Draw a diagonal blue line with thickness of 5 px
# 8 cv2.line(img,(0,0),(511,511),(255,0,0),5)
'''
import numpy as np
#  draw line  draw rectangle
img = np.zeros((512,512,3),np.uint8)
cv2.line(img,(0,0),(200,200),(255,0,0),5)  # 1,图片 2 起点 3 终点 4 颜色  5 线条粗细
cv2.rectangle(img,(0,0),(100,100),(0,255,0),-1) # 1 原图 2 左上角顶点 3 右下角顶点 4 颜色 5 线条粗细 如果为-1 则填充整个
cv2.circle(img, (256, 256), 50, (0, 0, 255), -1) # 1 圆中心点 2 半径 3 颜色 4 同上
cv2.ellipse(img,(320,320),(100,50),90,0,360,(30,30,30),-1) # 1 椭圆圆心 2 长轴和短轴大小 3 椭圆的旋转角度 比如水平的和垂直的 4 沿逆时针方向旋转 5 沿顺时针方向旋转
pts=np.array([[[325,325],[511,511],[511,400],[460,200]]], np.int32) # 多边形
print(pts.shape)
# pts=pts.reshape((-1,1,2))
a = np.array([[[10,10], [100,10], [100,100], [10,100]]], dtype = np.int32)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)
# cv2.polylines(img,pts,0,(255,255,255),50)
cv2.polylines(img, pts,1, 255,5)
cv2.imshow('drawline',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import cv2
import numpy as np #mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event==cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
# # 创建图像与窗口并将窗口与回调函数绑定 i
# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# events=[i for i in dir(cv2) if 'EVENT'in i]
# print(events)
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20)&0xFF==27:
#         break
# cv2.destroyAllWindows()


import cv2
import numpy as np
# 当鼠标按下时变为 True
drawing=False
# 如果 mode 为 true 绘制矩形。按下'm' 变成绘制曲线。
mode=True

ix,iy=-1,-1
# 创建回调函数

def draw_circle(events,x,y,flag,param):
    global ix,iy,drawing,mode
    if events == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif events == cv2.EVENT_MOUSEMOVE and flag == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
    # 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
                cv2.circle(img,(x,y),3,(0,0,255),-1)
    elif events == cv2.EVENT_LBUTTONUP:
        drawing = False


# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow("image")
# cv2.setMouseCallback('image',draw_circle) # 回调函数必须设置五个参数，否则会报错
# while True:
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1)
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break


def nothing(x): pass
# 创建一副黑色图像
img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    s=cv2.getTrackbarPos(switch,'image')
    if s==0:
        img[:]=0
    else:
        img[:]=[b,g,r]
cv2.destroyAllWindows()


'''
# 将鼠标控制画图和滑动条合并，一个可以控制画笔颜色的画板
import cv2
import numpy as np


def nothing(x): pass


# 当鼠标按下时变为 True
drawing = False
# 如果 mode 为 true 绘制矩形。按下'm' 变成绘制曲线。
mode=True
ix, iy = -1, -1


# 创建回调函数
def draw_circle(event, x, y, flags, param):
#
#
# # #
# # # #
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)
    global ix, iy, drawing, mode
# 当按下左键是返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
# 当鼠标左键按下并移动是绘制图形。event
# 可以查看移动，flag
# 查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
        # 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细 cv2.circle(img,(x,y),3,color,-1)
        # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
                cv2.circle(img, (x, y), 3, color, -1)
    # 当鼠标松开停止绘画
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False

img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

'''











