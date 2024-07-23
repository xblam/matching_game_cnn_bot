import cv2
import numpy as np

# Define the size of each tile
tile_size = 50  # Width and height of each image tile

# Create a dictionary to store images for each number
image_dict = {
    1: cv2.imread('1.png'),
    2: cv2.imread('2.png'),
    3: cv2.imread('3.png'),
    4: cv2.imread('4.png'),
    5: cv2.imread('5.png'),
    14: cv2.imread('14.png')
}

# Resize all images to the same size
for key in image_dict:
    image_dict[key] = cv2.resize(image_dict[key], (tile_size, tile_size))

# The input matrix
matrix = np.array([
    [14, 14, 2, 4, 3, 1, 4, 2, 4],
    [14, 14, 4, 3, 1, 2, 1, 3, 3],
    [3, 4, 1, 5, 2, 4, 1, 2, 5],
    [5, 5, 4, 5, 2, 5, 5, 4, 4],
    [4, 1, 2, 3, 1, 2, 3, 4, 2],
    [4, 1, 4, 4, 2, 4, 1, 3, 4],
    [2, 4, 3, 3, 5, 5, 4, 1, 2],
    [1, 2, 1, 1, 3, 3, 1, 4, 1],
    [4, 1, 3, 2, 1, 2, 1, 5, 2],
    [3, 2, 1, 2, 4, 2, 3, 2, 1]
])

# Determine the size of the canvas
rows, cols = matrix.shape
canvas_height = rows * tile_size
canvas_width = cols * tile_size

# Create a blank canvas
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Iterate through the matrix and place images
for i in range(rows):
    for j in range(cols):
        number = matrix[i, j]
        if number in image_dict:
            image = image_dict[number]
            x = j * tile_size
            y = i * tile_size
            canvas[y:y+tile_size, x:x+tile_size] = image

# Display the resulting image
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()