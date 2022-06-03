# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from cv2 import cv2
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("Your OpenCV version is: " + cv2.__version__)
    # a = np.array([[1, 0], [0, 1]])
    # b = np.array([[4, 1], [2, 2]])
    # print((a * b).sum())

    # A = np.array([[279, 552, 1, 0, 0, 0, -6696, -13248, -24],
    #               [0, 0, 0, 279, 552, 1, -157914, -312432, -566],
    #               [372, 559, 1, 0, 0, 0, -42408, -63726, -114],
    #               [0, 0, 0, 372, 559, 1, -205344, -308568, -552],
    #               [362, 472, 1, 0, 0, 0, -38372, -50032, -106],
    #               [0, 0, 0, 362, 472, 1, -171588, -223728, -474],
    #               [277, 469, 1, 0, 0, 0, -5263, -8911, -19],
    #               [0, 0, 0, 277, 469, 1, -133237, -225589, -481]])

    x_1 = [93, -7]
    y_1 = [63, 0]
    x_2 = [293, 3]
    y_2 = [868, -6]
    x_3 = [1207, 7]
    y_3 = [998, -4]
    x_4 = [1218, 3]
    y_4 = [309, 2]
    A = np.array([
        [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0] * x_1[1], y_1[0] * x_1[1], x_1[1]],
        [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0] * y_1[1], y_1[0] * y_1[1], y_1[1]],
        [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0] * x_2[1], y_2[0] * x_2[1], x_2[1]],
        [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0] * y_2[1], y_2[0] * y_2[1], y_2[1]],
        [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0] * x_3[1], y_3[0] * x_3[1], x_3[1]],
        [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0] * y_3[1], y_3[0] * y_3[1], y_3[1]],
        [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0] * x_4[1], y_4[0] * x_4[1], x_4[1]],
        [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0] * y_4[1], y_4[0] * y_4[1], y_4[1]],
    ])
    [U, S, Vt] = np.linalg.svd(A)
    homography = Vt[-1].reshape(3, 3)
    print(homography)
    transformedPoint = homography @ np.array([1679, 128, 1]).transpose()
    print(transformedPoint / transformedPoint[-1])  # will be ~[4, 7, 1]
