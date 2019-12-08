import numpy as np

GAUSSIAN_FILTER = np.array([[1, 1, 2, 2, 2, 1, 1],
                            [1, 2, 2, 4, 2, 2, 1],
                            [2, 2, 4, 8, 4, 2, 2],
                            [2, 4, 8, 16, 8, 4, 2],
                            [2, 2, 4, 8, 4, 2, 2],
                            [1, 2, 2, 4, 2, 2, 1],
                            [1, 1, 2, 2, 2, 1, 1]])


def gaussian_filtering(input_image):  # input_image ->  matrix containing pixel values
    Height1, width1, Height2, Width2 = input_image.shape[0], input_image.shape[1], 7, 7
    Height3 = Height2 // 2
    Width3 = Width2 // 2
    image_convolution = np.zeros((Height1, width1))

    Height4 = image_convolution.shape[0]
    Width4 = image_convolution.shape[1]

    # simple convolution operation
    i = Height3

    while i < Height4 - Height3:
        j = Width3
        while j < Width4 - Width3 :
            t = 0
            while t < Height2:
                k = 0
                while k < Width2:
                    image_convolution[i, j] = image_convolution[i, j] + \
                                     (input_image[i - Height3 + t, j - Width3 + k] * GAUSSIAN_FILTER[t, k])
                    k += 1
                t += 1
            j += 1
        i += 1

    # returning normalized gaussian smoothing
    return image_convolution / 140, Height3, Width3
