#coding:utf-8

import numpy as np
import cv2


class mouseParam:
    def __init__(self, input_img_name):
        # マウス入力用のパラメータ
        self.mouseEvent = {"x": None, "y": None, "event": None, "flags": None}
        # マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    # コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

        # マウス入力用のパラメータを返すための関数

    def getData(self):
        return self.mouseEvent

    # マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

        # マウスフラグを返す関数

    def getFlags(self):
        return self.mouseEvent["flags"]

        # xの座標を返す関数

    def getX(self):
        return self.mouseEvent["x"]

        # yの座標を返す関数

    def getY(self):
        return self.mouseEvent["y"]

        # xとyの座標を返す関数

    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


if __name__ == "__main__":
    # 入力画像
    read = cv2.imread("input.bmp")

    # 表示するWindow名
    window_name = "input window"

    # 画像の表示
    cv2.imshow(window_name, read)

    # コールバックの設定
    mouseData = mouseParam(window_name)

    while 1:
        cv2.waitKey(20)
        # 左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            file = open('point_x.txt', 'w')
            file.write(str(mouseData.getX()))
            file.close()
            file = open('point_y.txt', 'w')
            file.write(str(mouseData.getY()))
            file.close()
        # 右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONUP:
            break;

    cv2.destroyAllWindows()

#画像の読み込み
test = cv2.imread("input.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
width = test.shape[0]
height = test.shape[1]
frag = np.zeros(gray_test.shape)#領域分割フラグ

#画像の書き出し
cv2.imwrite('test.bmp', test)
cv2.imwrite('gray_test.bmp',gray_test)

#初期地点
file_data = open('point_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
cv_x = int(lines_x)

file_data = open('point_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
cv_y = int(lines_y)



color = 255
i = 0

stack = [cv_x,cv_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test[pyy][pxx] == 255):
        frag[pyy][pxx] = 255
        gray_test[pyy][pxx] = 0
        if ((pyy+1 < height) & (gray_test[pyy+1][pxx] == color)):
            stack.append(pxx)
            stack.append(pyy+1)
        if ((pxx+1 < width) & (gray_test[pyy][pxx+1] == color)):
            stack.append(pxx+1)
            stack.append(pyy)
        if (pyy-1 >= 0) & (gray_test[pyy-1][pxx] == color):
            stack.append(pxx)
            stack.append(pyy-1)
        if (pxx-1 >= 0) & (gray_test[pyy][pxx-1] == color):
            stack.append(pxx-1)
            stack.append(pyy)
        i += 1
print(i)
print("Finished")
cv2.imwrite('result.jpg', frag)

# カラー画像への処理
if __name__ == "__main__":
    # 入力画像
    read = cv2.imread("input_c.bmp")

    # 表示するWindow名
    window_name = "input_c window"

    # 画像の表示
    cv2.imshow(window_name, read)

    # コールバックの設定
    mouseData = mouseParam(window_name)

    print("色を二色選んでください\n色A:左クリック\n色B:右クリック\n終了:ホイール押し込み")
    while 1:
        cv2.waitKey(20)
        # 左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            file = open('color_a_x.txt', 'w')
            file.write(str(mouseData.getX()))
            file.close()
            file = open('color_a_y.txt', 'w')
            file.write(str(mouseData.getY()))
            file.close()
        # 右クリックで差表表示＆色取得
        if mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            print(mouseData.getPos())
            file = open('color_b_x.txt', 'w')
            file.write(str(mouseData.getX()))
            file.close()
            file = open('color_b_y.txt', 'w')
            file.write(str(mouseData.getY()))
            file.close()
        # Mボタンクリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONUP:
            break;

    cv2.destroyAllWindows()

# 座標の書き出し
#色A
file_data = open('color_a_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
ca_x = int(lines_x)

file_data = open('color_a_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
ca_y = int(lines_y)

# 色B
file_data = open('color_b_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
cb_x = int(lines_x)

file_data = open('color_b_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
cb_y = int(lines_y)

# 画像の読み込み
test_c = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test2 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag2 = np.zeros(test_c.shape)#領域分割フラグ

# 画像の書き出し
cv2.imwrite('test_c.bmp', test_c)

stack = [cv_x, cv_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test2[pyy][pxx] == 255):
        gray_test2[pyy][pxx] = 0
        if (set(test_c[pyy][pxx]) == set(test_c[ca_y][ca_x])):
            frag2[pyy][pxx] = [255, 0, 255]
            if ((pyy+1 < height) & (gray_test2[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test2[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test2[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test2[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)
        if (set(test_c[pyy][pxx]) == set(test_c[cb_y][cb_x])):
            frag2[pyy][pxx] = [255, 255, 0]
            if ((pyy+1 < height) & (gray_test2[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test2[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test2[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test2[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("Finished")
cv2.imwrite('result2.png', frag2)

# 色領域A
# 画像の読み込み
color_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test3 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag3 = np.zeros(color_1.shape)#領域分割フラグ

stack = [ca_x, ca_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test3[pyy][pxx] == 255):
        gray_test3[pyy][pxx] = 0
        if (set(color_1[pyy][pxx]) == set(color_1[ca_y][ca_x])):
            frag3[pyy][pxx] = [255, 255, 255]
            if ((pyy+1 < height) & (gray_test3[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test3[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test3[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test3[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("色A領域")
cv2.imwrite('color_1.png', frag3)

# 膨張
di_1 = cv2.imread('color_1.png', 0)
kernel = np.ones((5, 5), np.uint8)
cv2.imwrite('di_1.png', cv2.dilate(di_1, kernel, iterations = 1))

# 色A領域(元画像)
di_c1 = cv2.imread('color_1.png', 0)
# 膨張画像
di_a1 = cv2.imread('di_1.png', 0)
# 結果反映用画像
di_r1 = cv2.imread('di_1.png', 0)
for i in range(height):
    for j in range(width):
        if (di_c1[i][j] == di_a1[i][j]):
            di_r1[i][j] = 0

# 膨張画像di_1と色領域Aの差分
cv2.imwrite('gradient_a.png', di_r1)


# 色領域B
# 画像の読み込み
color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test4 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag4 = np.zeros(color_2.shape)#領域分割フラグ

stack = [cb_x, cb_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test4[pyy][pxx] == 255):
        gray_test4[pyy][pxx] = 0
        if (set(color_2[pyy][pxx]) == set(color_1[cb_y][cb_x])):
            frag4[pyy][pxx] = [255, 255, 255]
            if ((pyy+1 < height) & (gray_test4[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test4[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test4[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test4[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("色B領域")
cv2.imwrite('color_2.png', frag4)

# 膨張
di_2 = cv2.imread('color_2.png', 0)
cv2.imwrite('di_2.png', cv2.dilate(di_2, kernel, iterations = 1))

# 色A領域(元画像)
di_c2 = cv2.imread('color_2.png', 0)
# 膨張画像
di_a2 = cv2.imread('di_2.png', 0)
# 結果反映用画像
di_r2 = cv2.imread('di_2.png', 0)
for i in range(height):
    for j in range(width):
        if (di_c2[i][j] == di_a2[i][j]):
            di_r2[i][j] = 0

# 膨張画像di_2と色領域Bの差分
cv2.imwrite('gradient_b.png', di_r2)

# 膨張Aと色B領域の重なり部分の処理
# 膨張差分A
gr_a = cv2.imread('gradient_a.png', 0)
# 色B領域
cregion_b = cv2.imread('color_2.png', 0)
# 反映先画像
tile_a = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((gr_a[i][j] == 255) & (cregion_b[i][j] == 255)):
            tile_a[i][j] = 255

# タイルパターン領域aの生成
cv2.imwrite('tile_a.png', tile_a)

# 膨張Bと色A領域の重なり部分の処理
# 膨張差分B
gr_b = cv2.imread('gradient_b.png', 0)
# 色B領域
cregion_a = cv2.imread('color_1.png', 0)
# 反映先画像
tile_b = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((gr_b[i][j] == 255) & (cregion_a[i][j] == 255)):
            tile_b[i][j] = 255

# タイルパターン領域bの生成
cv2.imwrite('tile_b.png', tile_b)



# タイルパターン領域aとbの合成
# 膨張差分B
tile_a2 = cv2.imread('tile_a.png', 0)
# 色B領域
tile_b2 = cv2.imread('tile_b.png', 0)
# 反映先画像
tile_ab = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if (tile_a2[i][j] != tile_b2[i][j]):
            tile_ab[i][j] = 255

# タイルパターン領域bの生成
cv2.imwrite('tile_ab.png', tile_ab)