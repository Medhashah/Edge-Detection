import os
import cv2
import Thresholding
import GuassianFilter
import SobelOperator

# static path for storting images
PATH = "Output/"


def canny_edge_detector(input, source, e):
    # creating folders if not created previously
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    if not os.path.exists(PATH + input):
        os.makedirs(PATH + input)

    Image_after_filter, H1, W1 = GuassianFilter.gaussian_filtering(source)  # compute gaussian filter
    cv2.imwrite(PATH + input + "/gaussianblur.bmp", Image_after_filter)  # save gaussian image.

    x, y = SobelOperator.sobel(Image_after_filter, H1, W1)  # find horizontal gradient
    cv2.imwrite(PATH + input + "/horizontalgradient.bmp", x)  # save horizontal gradient
    cv2.imwrite(PATH + input + "/verticalgradient.bmp", y)  # save vertical gradient

    # compute gradient magnitude and angle
    gradient_magnitude, gradient_angle = Thresholding.compute_magnitude_angle(x, y)
    cv2.imwrite(PATH + input + "/gradientmagnitude.bmp", gradient_magnitude)  # save gradient_magnitude

    # non maximuma supression on gradient magnitude.
    c = Thresholding.suppression(gradient_magnitude, gradient_angle)
    cv2.imwrite(PATH + input + "/nonmaximasuppression.bmp", c)

    e = Thresholding.doublethresholding(c, gradient_angle)
    cv2.imwrite(PATH + input + "/doublethresholding.bmp", e) # save after doublethresholding

