import numpy as np
import matplotlib.pyplot as plt


# https://github.com/davechristian/Simple-SSD-Stereo/blob/main/stereomatch_SSD.py
# https://github.com/2b-t/stereo-matching
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
            best_ssd = float('inf')  # init as max error (max float)
            for offset in range(disp_range[0], disp_range[1]):
                ssd = error_func(img_l, img_r, y, x, offset, k_size)
                if ssd < best_ssd:
                    best_ssd = ssd
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
    # l_mean, r_mean, n = 0, 0, 0
    # # Loop over window
    # for v in range(-ker_half, ker_half + 1):
    #     for u in range(-ker_half, ker_half + 1):
    #         # Calculate cumulative sum
    #         l_mean += img_l[y + v, x + u]
    #         r_mean += img_r[y + v, x + u - offset]
    #         n += 1
    # l_mean = l_mean / n
    # r_mean = r_mean / n

    # if the window exceeds the limits of the image return max error
    if x - ker_half - offset < 0:
        return float('inf')
    l_win = img_l[y - ker_half: y + ker_half + 1, x - ker_half: x + ker_half + 1]
    r_win = img_r[y - ker_half: y + ker_half + 1, x - ker_half - offset: x + ker_half + 1 - offset]
    l_mean = np.mean(l_win)
    r_mean = np.mean(r_win)
    l_win = l_win - l_mean
    r_win = r_win - r_mean
    # l_r, l_var, r_var = 0, 0, 0
    # for v in range(-ker_half, ker_half + 1):
    #     for u in range(-ker_half, ker_half + 1):
    #         # Calculate terms
    #         l = img_l[y + v, x + u] - l_mean
    #         r = img_r[y + v, x + u - offset] - r_mean
    #         l_r += l * r
    #         l_var += l ** 2
    #         r_var += r ** 2
    l_r = (l_win * r_win).sum()
    l_var = (l_win * l_win).sum()
    r_var = (r_win * r_win).sum()
    # Assemble terms
    # TODO: why minus?
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
    pass


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

    ##### Your Code Here ######

    # out = dst_img * mask + src_out * (1 - mask)
    # plt.imshow(out)
    # plt.show()
