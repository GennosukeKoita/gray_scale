import sys
import cv2
import numpy as np
'''
_, infile, outfile = sys.argv

img_bgr = cv2.imread(infile)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) #グレースケール化

cv2.imwrite(outfile, img_gray) #出力
'''
'''
gamma22LUT = np.array([pow(x/255.0, 2.2) * 255 for x in range(256)], dtype='uint8')
gamma045LUT = np.array([pow(x/255.0, 1.0 / 2.2) * 255 for x in range(256)], dtype='uint8')

print(gamma22LUT)
print(gamma045LUT)

_, infile, outfile = sys.argv

img_bgr = cv2.imread(infile) #入力画像読み込み
img_bgrL = cv2.LUT(img_bgr, gamma22LUT)

img_grayL = cv2.cvtColor(img_bgrL, cv2.COLOR_BGR2GRAY)
img_gray = cv2.LUT(img_grayL, gamma045LUT)

cv2.imwrite(outfile, img_gray) #出力
'''
img = cv2.imread('scale.jpg')
print(img.shape) #（ピクセル数＊ピクセル数＊3(RGB)）
def numpy_gray(src):
    r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    return gray

gray_num = numpy_gray(img)
print(gray_num)
cv2.imwrite("out_scale.jpg", gray_num)
