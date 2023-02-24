import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

# The project's goal: Determine the best k lines, in terms of quality and length
# To determine the best top k lines in terms of length and quality we
# would like to apply edge detection using the Sobel filter.
# Then apply Hough transform to detect and rank the lines.


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel)


def convolution(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output


def apply_edge_detection(image, filter_x, filter_y):
    new_image_x = convolution(image, filter_x)
    new_image_y = convolution(image, filter_y)
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_direction = np.arctan2(new_image_y, new_image_x)

    return gradient_magnitude, gradient_direction


# Edge thinning
def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)

    pi = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < pi / 8) or (15 * pi / 8 <= direction <= 2 * pi):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (pi / 8 <= direction < 3 * pi / 8) or (9 * pi / 8 <= direction < 11 * pi / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * pi / 8 <= direction < 5 * pi / 8) or (11 * pi / 8 <= direction < 13 * pi / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    return output


def threshold(image, low, high):
    output = np.zeros(image.shape)
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    output[strong_row, strong_col] = 255
    output[weak_row, weak_col] = 0

    return output


def black_or_white(image, row, col):
    if image[row, col + 1] == 255 or image[row, col - 1] == 255 or image[
        row - 1, col] == 255 or image[row + 1, col] == 255 or image[
        row - 1, col - 1] == 255 or image[row + 1, col - 1] == 255 or image[
        row - 1, col + 1] == 255 or image[row + 1, col + 1] == 255:
        return 255
    return 0


def hysteresis(image, white):
    image_row, image_col = image.shape

    for row in range(1, image_row):
        for col in range(1, image_col):
            if image[row, col] == white:
                image[row, col] = black_or_white(image, row, col)
    return image


def hough_transform(img):
    thetas = []
    thetas_range = 90
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    accumulator = np.zeros((2 * diag_len, thetas_range*2), dtype=np.uint8)
    y_s, x_s = np.nonzero(img == 255)

    for i in range(len(x_s)):
        x = x_s[i]
        y = y_s[i]

        for index in range(thetas_range*2):
            #convert to radian
            theta = math.radians(index-thetas_range)
            rho = diag_len + int(round(x * math.cos(theta) + y * math.sin(theta)))
            accumulator[rho, index] += 1
            thetas.append(theta)

    return accumulator, thetas, rhos



def find_best_k_lines(accumulator, thetas, rhos, k):
    threshold = 0.2 * np.max(accumulator)
    peaks = np.argwhere(accumulator > threshold)
    # Sort the peaks by accumulator value in descending order
    peaks = peaks[np.argsort(accumulator[peaks[:, 0], peaks[:, 1]])][::-1]
    lines = []
    for i, peak in enumerate(peaks):
        if i == k:
            break
        rho = rhos[peak[0]]
        theta = thetas[peak[1]]
        lines.append((rho, theta))

    return lines


def draw_lines(image, lines):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (160, 255, 0), 1)
    return image


def draw_lines_on_image(image, accumulator, thetas, rhos, k):
    lines = find_best_k_lines(accumulator, thetas, rhos, k)
    # Draw the lines on a blank image
    lines_image = draw_lines(np.zeros_like(image), lines)
    # Overlay the lines on the original image
    result = cv2.addWeighted(image, 0.8, lines_image, 1, 0)
    return result


def edge_detection(image):
    blurred_image = gaussian_blur(image, kernel_size=5)
    edge_filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edge_filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_magnitude, gradient_direction = apply_edge_detection(blurred_image, edge_filter_x, edge_filter_y)

    new_image = non_max_suppression(gradient_magnitude, gradient_direction)
    new_image = threshold(new_image, 5, 25)
    new_image = hysteresis(new_image, 255)

    return new_image


def detect_best_k_lines(image_path, k):
    image_grey = cv2.imread(image_path, 0)
    image = cv2.imread(image_path)
    edges = edge_detection(image_grey)
    cv2.imwrite("edges.jpg", edges)
    accumulator, thetas, rhos = hough_transform(edges)
    image_with_lines = draw_lines_on_image(image, accumulator, thetas, rhos, k)
    cv2.imwrite("result.jpg", image_with_lines)


if __name__ == '__main__':
    path = input("please enter the image path ")
    k = int(input("please enter how many lines you want to draw "))
    detect_best_k_lines(path, k)
