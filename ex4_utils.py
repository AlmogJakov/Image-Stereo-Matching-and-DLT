import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    return disparityCalc(img_l, img_r, disp_range, k_size, computeSSD)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    return disparityCalc(img_l, img_r, disp_range, k_size, computeNCC)


def disparityCalc(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int, error_func) -> np.ndarray:
    w, h = img_l.shape  # assume that both images are same size
    depth = np.zeros((w, h), np.float32)
    depth.shape = h, w
    kernel_half = int(k_size / 2)
    for y in range(kernel_half, h - kernel_half):
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            lowest_error = float('inf')  # init as max error (max float)
            for offset in range(disp_range[0], disp_range[1]):
                error = error_func(img_l, img_r, y, x, offset, k_size)
                if error < lowest_error:
                    lowest_error = error
                    best_offset = offset
            depth[y, x] = best_offset / 255
    return depth


def computeSSD(img_l: np.ndarray, img_r: np.ndarray, y: int, x: int, offset: int, k_size: int):
    kernel_half = int(k_size / 2)
    ssd = 0
    for v in range(-kernel_half, kernel_half + 1):
        for u in range(-kernel_half, kernel_half + 1):
            ssd_temp = (img_l[y + v, x + u] - img_r[y + v, x + u - offset]) ** 2
            ssd += ssd_temp * ssd_temp
    return ssd


def computeNCC(img_l: np.ndarray, img_r: np.ndarray, y: int, x: int, offset: int, k_size: int):
    ker_half = int(k_size / 2)
    # if the window exceeds the limits of the image return max error
    if x - ker_half - offset < 0:
        return float('inf')
    l_win = img_l[y - ker_half: y + ker_half + 1, x - ker_half: x + ker_half + 1]
    r_win = img_r[y - ker_half: y + ker_half + 1, x - ker_half - offset: x + ker_half + 1 - offset]
    l_mean = np.mean(l_win)
    r_mean = np.mean(r_win)
    l_win = l_win - l_mean
    r_win = r_win - r_mean
    l_r = (l_win * r_win).sum()
    l_var = (l_win * l_win).sum()
    r_var = (r_win * r_win).sum()
    # Assemble terms
    # NCC returns a value between -1 and +1.
    #   Near 0 indication that there is no correlation.
    #   Near +1 means, that the image is very similar to the other one.
    #   Near -1 means, that it's likely that one image is a negative and should be inverted.
    # Source: https://stackoverflow.com/questions/17189513/does-a-negative-cross-correlation-show-high-or-low-similarity
    # Since we treat this method as an error function then we return the inverse value (minus)
    # Therefore for the best match (value 1) we get a minimum error (value -1)
    return -l_r / np.sqrt(l_var * r_var)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    a_matrix = []
    for i in range(src_pnt.shape[0]):
        x_src, y_src = src_pnt[i][0], src_pnt[i][1]
        x_dst, y_dst = dst_pnt[i][0], dst_pnt[i][1]
        a_matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst])
        a_matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src, -y_dst])
    [u, s, vt] = np.linalg.svd(np.array(a_matrix))
    # We need the last column of v^t so we take last row of v.
    # we can get the matrix with:
    #   homography = Vt[-1].reshape(3, 3)
    # but we want the last number to be 1 (homography[2][2] = 1)
    # so we need to divide homography matrix by the last number (/ homography[2, 2])
    homography = (vt[-1]).reshape(3, 3)
    homography = homography / homography[2, 2]
    total_error = 0
    for i in range(src_pnt.shape[0]):
        src_points = np.append(src_pnt[i], 1)
        dst_points = np.append(dst_pnt[i], 1)
        res = homography.dot(src_points)
        res_homogeneous = res / res[2]
        total_error += np.sqrt(sum(res_homogeneous - dst_points) ** 2)
    return homography, total_error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))
        plt.plot(x, y, '*r')
        dst_p.append([x, y])
        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    # ##### Your Code Here ######
    src_p = []
    fig2 = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))
        plt.plot(x, y, '*r')
        src_p.append([x, y])
        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    homography, total_error = computeHomography(src_p, dst_p)
    homography = np.array(homography)
    src_out = cv2.warpPerspective(src_img, np.array(homography), (dst_img.shape[1], dst_img.shape[0]))
    mask = np.array((src_out == [0, 0, 0]).all(-1), dtype=np.float32)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = dst_img * mask + src_out * (1 - mask)
    plt.imshow(out)
    plt.show()