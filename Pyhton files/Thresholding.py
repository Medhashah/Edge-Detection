import numpy as np
import math


def sector(angle):  # angle -> gradient angle

    if angle < 0:
        angle = angle + 360
    if 0 <= angle <= 22.5 or 157.5 < angle <= 202.5 or 337.5 < angle <= 360:
        return 0
    elif 22.5 < angle <= 67.5 or 202.5 < angle <= 247.5:
        return 1
    elif 67.5 < angle <= 112.5 or 247.5 < angle <= 292.5:
        return 2
    elif 112.5 < angle <= 157.5 or 292.5 < angle <= 337.5:
        return 3

    # return sector -> gives sector according to gradient angle of the center pixel.


def compute_magnitude_angle(gx, gy): # gx -> horizontal gradient,  gy -> vertical gradient

    magnitude = np.zeros((gx.shape[0], gx.shape[1]))
    angle = np.zeros((gx.shape[0], gx.shape[1]))

    i = 0

    while i < gx.shape[0]:
        j = 0
        while j < gx.shape[1]:
            magnitude[i, j] = math.sqrt((gx[i, j] * gx[i, j]) + (gy[i, j] * gy[i, j]))

            if gx[i, j] == 0:
                if gy[i, j] > 0:
                    angle[i, j] = 90.0
                else:
                    angle[i, j] = -90.0
            else:
                angle[i, j] = math.degrees(math.atan(gy[i, j] / gx[i, j]))
            j += 1
        i += 1

    return magnitude / 1.4142, angle


def suppression(gradient_magnitude, gradient_angle):
#gradient_magnitude -> holds gradient magnitude of every pixel,gradient_angle ->  holds gradient angle of every pixel.

    height = gradient_magnitude.shape[0]
    width = gradient_magnitude.shape[1]
    non_maxima_suppression = np.zeros((height, width))
    i = 1
    while i < height - 1:
        j = 1
        while j < width - 1:
            sector_1 = sector(gradient_angle[i, j])
            if sector_1 == 0:
                maximum = max(gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1])
                if gradient_magnitude[i, j] > maximum:
                    non_maxima_suppression[i, j] = gradient_magnitude[i, j]
            elif sector_1 == 1:
                maximum = max(gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1])
                if gradient_magnitude[i, j] > maximum:
                    non_maxima_suppression[i, j] = gradient_magnitude[i, j]
            elif sector_1 == 2:
                maximum = max(gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j])
                if gradient_magnitude[i, j] > maximum:
                    non_maxima_suppression[i, j] = gradient_magnitude[i, j]
            elif sector_1 == 3:
                maximum = max(gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1])
                if gradient_magnitude[i, j] > maximum:
                    non_maxima_suppression[i, j] = gradient_magnitude[i, j]
            j += 1
        i += 1
    # return : image matrix -> gradient magnitude of those pixels which contributes in forming edge.
    return non_maxima_suppression


def doublethresholding(image, angle):
    # image -> matrix, ange -> gradient angle

    Height = image.shape[0]
    Width = image.shape[1]

    Value = np.amax(image)

    highRatio = 0.20
    Threshold1 = Value * highRatio

    lowRatio = 0.10
    Threshold2 = Value * lowRatio

    print(Threshold1,Threshold2)

    num = 255

    Output = np.empty((Height, Width), dtype='int')

    strongpixel = image > Threshold1

    # strongpixels are a part of the edge
    Output[strongpixel] = num

    nopixel= image < Threshold2

    # nopixels are a part of the edge
    Output[nopixel] = 0

    weakpixel = (image >= Threshold2) & (image <= Threshold1)

    indexweakpixel = np.argwhere(weakpixel)

    for i in indexweakpixel:

        x = i[0]
        y = i[1]

        gradient = angle[x][y]

        if x > 0 and y > 0 and x < Width and y < Height:

            n = image[x - 1][y]
            ng = angle[x - 1][y]

            s = image[x + 1][y]
            sg = angle[x + 1][y]

            e = image[x][y + 1]
            eg = angle[x][y + 1]

            w = image[x][y - 1]
            wg = angle[x][y - 1]

            ne = image[x - 1][y + 1]
            neg = angle[x - 1][y + 1]

            nw = image[x - 1][y - 1]
            nwg = angle[x - 1][y - 1]

            sw = image[x + 1][y - 1]
            swg = angle[x + 1][y - 1]

            se = image[x + 1][y + 1]
            seg = angle[x + 1][y + 1]

            # check if any of the 8 neighbors is a strong edge pixel
            if (((n in strongpixel) and abs(ng -gradient) > 45) or ((s in strongpixel) and abs(sg-gradient) > 45) or ((e in strongpixel) and abs(eg-gradient) > 45) or ((
                    w in strongpixel) and abs(wg-gradient) > 45)
                    or ((ne in strongpixel)and abs(neg-gradient) > 45) or ((nw in strongpixel) and abs(nwg-gradient) > 45) or ((
                            se in strongpixel) and abs(seg-gradient) > 45) or
                    ((sw in strongpixel) and abs(swg-gradient) > 45)):
                # classify the pixel as an edge pixel
                Output[x][y] = num

    # return image after double thresholding
    return Output
