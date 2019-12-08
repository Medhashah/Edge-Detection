import numpy as np

SOBELOPERATOR_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

SOBELOPERATOR_GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def sobel(image, height, width):
	sobeloperator_gx = convolution(image, SOBELOPERATOR_GX, height, width)
	sobeloperator_gy = convolution(image, SOBELOPERATOR_GY, height, width)
	return sobeloperator_gx,sobeloperator_gy


def convolution(image, g, height, width):
	# Image -> Matrix , g  -> kernel 3*3 sobel operator , height -> gaussian image height, width ->  gaussian image width

	rows, columns = image.shape
	image_convolution = np.zeros(image.shape)
	Height1 = 1
	Width1	= 1

	i = height+1
	while i < rows - (height + 1):
		j = width + 1
		while j < columns - (width + 1):
			image_convolution[i,j] = 0
			t = -Height1
			while t < Height1 + 1:
				k = -Width1
				while k < Width1 + 1:
					image_convolution[i,j] = image_convolution[i,j] + g[Height1+t,Width1+k] * image[i + t, j + k]
					k += 1
				t += 1
			if image_convolution[i,j] < 0:
				# taking absolute value
				image_convolution[i,j] = abs(image_convolution[i, j])

			# normalizing gradients
			image_convolution[i,j] = image_convolution[i, j] / 3.0
			j += 1
		i += 1


	return image_convolution




