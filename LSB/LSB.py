import numpy as np ,cv2
import matplotlib.pyplot as plt


Aimage=cv2.imread("./test3.jpg",cv2.IMREAD_COLOR)
if Aimage is None: raise Exception("영상파일 오류")
Bimage=cv2.imread("./B.jpg",cv2.IMREAD_GRAYSCALE)
if Bimage is None: raise Exception("영상파일 오류")

A_copy=Aimage.copy()
B_copy=Bimage.copy()

row,col=Aimage.shape[0],Aimage.shape[1]

#B image Resize , Threshold
B_copy=cv2.resize(B_copy,(1920,1280))
_,B_copy=cv2.threshold(B_copy,44.9,255,cv2.THRESH_BINARY)
plt.hist(B_copy.ravel(), 256, [0,256])
plt.show()

#Blue 채널 LSB에 Binary image B를 기록
for i in range(row):
    for j in range(col):
        if B_copy[i][j] ==0:
            if A_copy[i][j][0]%2 !=0:
                A_copy[i][j][0]=A_copy[i][j][0]-1
        else:
            if A_copy[i][j][0] % 2 == 0:
                A_copy[i][j][0]= A_copy[i][j][0]+ 1

B_p=B_copy.copy()

## image B' 재구성
for i in range(row):
    for j in range(col):
        if A_copy[i][j][0]%2==0:
            B_p[i][j]=0
        else:
            B_p[i][j]=255
# image A'의 LSB값을 이진이미지로 출력
A_p=A_copy.copy()
for i in range(row):
    for j in range(col):
        if A_copy[i][j][0]%2==0:
            A_p[i][j]=0
        else:
            A_p[i][j]=255


#Image A 이진이미지
AA=B_copy.copy()
for i in range(row):
    for j in range(col):
        if Aimage[i][j][0]%2==0:
            AA[i][j]=0
        else:
            AA[i][j]=255

# BGR 을 RGB로
Aimage=cv2.cvtColor(Aimage,cv2.COLOR_BGR2RGB)
A_copy=cv2.cvtColor(A_copy,cv2.COLOR_BGR2RGB)


def binaryclassfier(img,testimg,row,col):
    #채널 별 히스토그램 보니 LSB 이미지 blue channel 주변 빈도수 차이가 큼

    s1,s2=0,0
    A_even, A_odd = len(np.where(Aimage.flatten() % 2 == 0)[0]), len(np.where(Aimage.flatten() % 2 != 0)[0])
    Ap_even, Ap_odd = len(np.where(A_copy.flatten() % 2 == 0)[0]), len(np.where(A_copy.flatten() % 2 != 0)[0])
    even,odd=A_even / (A_even + A_odd) * 100,A_odd / (A_even + A_odd) * 100
    peven,podd=Ap_even / (Ap_even + Ap_odd) * 100,Ap_odd / (Ap_even + Ap_odd) * 100

    if even>peven and odd<podd:
        m=1
    else:
        m=2

    print(A_even / (A_even + A_odd) * 100, ':', A_odd / (A_even + A_odd) * 100)
    print(Ap_even / (Ap_even + Ap_odd) * 100, ':', Ap_odd / (Ap_even + Ap_odd) * 100)

    hist1 = cv2.calcHist([img], [2], None, [256], [0, 256])
    hist2= cv2.calcHist([testimg], [2], None, [256], [0, 256])
    plt.plot(hist2, color="blue")
    plt.xlim([0, 256])
    plt.show()
    if m== 1:
        print("홀수 픽셀 비율이 높을 때")
        for i in range(row):
            for j in range(col):
                if img[i][j][2]%2!=0:
                    s1+=1
        for i in range(row):
            for j in range(col):
                if testimg[i][j][2]%2!=0:
                    s2+=1
    else:
        print("짝수 픽셀 비율이 높을 때")
        for i in range(row):
            for j in range(col):
                if img[i][j][2]%2==0:
                    s1+=1
        for i in range(row):
            for j in range(col):
                if testimg[i][j][2]%2==0:
                    s2+=1
    print(s1)
    print(s2)
    if s1<s2:
        print("LSB img")
    else:
        print("not LSB img ")

# image ,test image
binaryclassfier(Aimage,A_copy,row,col)


plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1), plt.title("A_img"), plt.imshow(Aimage)
plt.subplot(2, 2, 2), plt.title("A_LSB_img"), plt.imshow(A_copy)
plt.subplot(2, 2, 3), plt.title("Image A binary"), plt.imshow(AA,cmap="gray")
plt.subplot(2, 2, 4), plt.title("Image A' binary"), plt.imshow(A_p)
plt.show()
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1), plt.title("A_Hist"),plt.hist(Aimage.flatten(), bins=list(range(0, 255)))
plt.subplot(1, 2, 2), plt.title("A'_Hist"),plt.hist(A_copy.flatten(), bins=list(range(0, 255)))
plt.show()
