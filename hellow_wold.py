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
        # Mクリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONUP:
            break;

    cv2.destroyAllWindows()

# 画像の読み込み
test = cv2.imread("input.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
width = test.shape[0]
height = test.shape[1]
frag = np.zeros(gray_test.shape)#領域分割フラグ

# 画像の書き出し
cv2.imwrite('test.bmp', test)
cv2.imwrite('gray_test.bmp',gray_test)

# 初期地点
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
    magnification = 1
    while 1:
        cv2.waitKey(20)
        # 左クリックで座標表示＆色取得
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            if magnification == 1:
                print(mouseData.getPos())
                file = open('color_a_x.txt', 'w')
                file.write(str(mouseData.getX()))
                file.close()
                file = open('color_a_y.txt', 'w')
                file.write(str(mouseData.getY()))
                file.close()
            else:
                print("(", mouseData.getX()//magnification, ",", mouseData.getY()//magnification, ")")
                file = open('color_a_x.txt', 'w')
                file.write(str(mouseData.getX()//magnification))
                file.close()
                file = open('color_a_y.txt', 'w')
                file.write(str(mouseData.getY()//magnification))
                file.close()
        # 右クリックで座標表示＆色取得
        if mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            if magnification == 1:
                print(mouseData.getPos())
                file = open('color_b_x.txt', 'w')
                file.write(str(mouseData.getX()))
                file.close()
                file = open('color_b_y.txt', 'w')
                file.write(str(mouseData.getY()))
                file.close()
            else:
                print("(", mouseData.getX()//magnification, ",", mouseData.getY()//magnification, ")")
                file = open('color_b_x.txt', 'w')
                file.write(str(mouseData.getX()//magnification))
                file.close()
                file = open('color_b_y.txt', 'w')
                file.write(str(mouseData.getY()//magnification))
                file.close()
        # 20Fescキー長押しで画像の2倍
        if cv2.waitKey(20) & 0xFF == 27:
                zoomed_image = read.repeat(magnification*2, axis=0).repeat(magnification*2, axis=1)
                cv2.imshow(window_name, zoomed_image)
                magnification = magnification*2
                cv2.waitKey(80)
        # Mボタンクリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONUP:
            break

    cv2.destroyAllWindows()

# 座標の書き出し
# 色A
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
kernel = np.ones((3, 3), np.uint8)
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

# 色B領域(元画像)
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

# タイルパターン領域abの生成
cv2.imwrite('tile_ab.png', tile_ab)

# 真ん中領域の膨張
# 膨張
center_b = cv2.imread('tile_ab.png', 0)
cv2.imwrite('center_b.png', cv2.dilate(center_b, kernel, iterations = 1))
center_bb = cv2.imread('center_b.png', 0)
# 結果反映用画像
center_ll = cv2.imread('center_b.png', 0)
# 膨張画像との差分
for i in range(height):
    for j in range(width):
        if (center_bb[i][j] == center_b[i][j]):
            center_ll[i][j] = 0

cv2.imwrite('center_ll.png', center_ll)


# 左領域(A)(di_c1)との共通部分
# 色A領域(元画像) di_c1 = cv2.imread('color_1.png', 0)
# 結果反映用画像
tile_left = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((center_ll[i][j] == 255) & (di_c1[i][j] == 255)):
            tile_left[i][j] = 255

# 膨張画像center_llと色領域A(di_c1=color_1.png)の重なり部分
cv2.imwrite('tile_left.png', tile_left)

# 右領域(B)(di_c2)との共通部分
# 色B領域(元画像) di_c2 = cv2.imread('color_2.png', 0)
# 結果反映用画像
tile_right = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((center_ll[i][j] == 255) & (di_c2[i][j] == 255)):
            tile_right[i][j] = 255

# 膨張画像center_llと色領域B(di_c2=color_2.png)の重なり部分
cv2.imwrite('tile_right.png', tile_right)


# タイルパターン領域abとright,leftの合成
# 真ん中center_b = cv2.imread('tile_ab.png', 0)
# 右領域
tile_br = cv2.imread('tile_right.png', 0)
# 左領域
tile_bl = cv2.imread('tile_left.png', 0)
# 反映先画像
tile_rabl = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if (tile_br[i][j] != center_b[i][j]):
            tile_rabl[i][j] = 255
        if (tile_bl[i][j] != center_b[i][j]):
            tile_rabl[i][j] = 255

# タイルパターン領域abの生成
cv2.imwrite('tile_rabl.png', tile_rabl)


# 局所探索領域の作成
# abとright,leftの合成画像
tile_rabl2 = cv2.imread('tile_rabl.png', 0)
# 縦方向走査
check_p = 0
count_s = 0
count_t = 0
for i in range(width):
    for j in range(height):
        if (tile_rabl2[j][i] == 255):
            if check_p == 0:
                left_x = i
                check_p = 1
                break
            else:
                right_x = i
                count_s += 1
                break
    if count_s != 0:
        count_t += 1
        if count_s != count_t:
            break

print(left_x, right_x)
# 横方向走査
check_p = 0
count_s = 0
count_t = 0
for i in range(height):
    for j in range(width):
        if (tile_rabl2[i][j] == 255):
            if check_p == 0:
                up_y = i
                check_p = 1
                break
            else:
                under_y = i
                count_s += 1
                break
    if count_s != 0:
        count_t += 1
        if count_s != count_t:
            break

print(up_y, under_y)

# 反映先画像
lsr = np.zeros(color_2.shape)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        lsr[i][j] = 255

# タイルパターンを施す長方形画像の生成
cv2.imwrite('lsr.png', lsr)


# 50%パターン
# 結果反映用画像lsr2
tile_50 = cv2.imread('lsr.png', 0)
if ((up_y + left_x) % 2) == 0:
    upper_left_c = 0
else:
    upper_left_c = 1
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if upper_left_c == 0:
            if ((i + j) % 2) == 1:
                tile_50[i][j] = 0
        else:
            if ((i + j) % 2) == 0:
                tile_50[i][j] = 0

# 50%確認用
cv2.imwrite('tile_50.png', tile_50)

# 50%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 真ん中(50%)のTRP center_b = cv2.imread('tile_ab.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
# 結果反映用画像finish_50
finish_50 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_50[i][j] == 255) & (center_b[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_50[i][j] == 0) & (center_b[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%確認用
cv2.imwrite('finish_50.png', finish_50)


# 25%パターン
# 結果反映用画像tile_25
tile_25 = cv2.imread('lsr.png', 0)
if ((up_y + left_x) % 2) == 0:
    upper_left_c = 0
else:
    upper_left_c = 1
Nol = 0
for i in range(up_y, under_y+1):
    if (Nol % 2) != 0:
        for j in range(left_x, right_x+1, 2):
                tile_25[i][j] = 0
    Nol += 1
# 25%確認用
cv2.imwrite('tile_25.png', tile_25)

# 50%タイル画像に25%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 左領域(25%)tile_bl = cv2.imread('tile_left.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_25[i][j] == 255) & (tile_bl[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_25[i][j] == 0) & (tile_bl[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%+25%適用確認用
cv2.imwrite('finish_50+25.png', finish_50)


# 75%パターン
# 結果反映用画像lsr2
tile_75 = cv2.imread('lsr.png', 0)
Nol =0
for i in range(up_y, under_y+1):
    if (Nol % 2) == 0:
        for j in range(left_x, right_x+1):
            if upper_left_c == 0:
                if ((i + j) % 2) == 1:
                    tile_75[i][j] = 0
            else:
                if ((i + j) % 2) == 0:
                    tile_75[i][j] = 0
    else:
        for j in range(left_x, right_x + 1):
            tile_75[i][j] = 0
    Nol += 1
# 50%確認用
cv2.imwrite('tile_75.png', tile_75)

# 50+25%タイル画像に75%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 右領域(75%)tile_br = cv2.imread('tile_right.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_75[i][j] == 255) & (tile_br[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_75[i][j] == 0) & (tile_br[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%+25%適用確認用
cv2.imwrite('finish_50+25+75.png', finish_50)