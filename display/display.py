import cv2
import numpy as np


# class DisplayGame:
# Define the size of each tile
tile_size = 50  # Width and height of each image tile


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

image_files = {
    1: r'./display/1.png',
    2: r'./display/2.png',
    3: r'./display/3.png',
    4: r'./display/4.png',
    5: r'./display/5.png',
    6: r'./display/6.png',
    7: r'./display/7.png',
    8: r'./display/8.png',
    9: r'./display/9.png',
    10: r'./display/10.png',
    11: r'./display/11.png',
    12: r'./display/12.png',
    13: r'./display/13.png',
    14: r'./display/14.png'
}


def display_matrix(matrix):
    # Create a dictionary to store images for each number
    

    
    image_dict = {}
    
    # Load and resize the images
    for key, file_name in image_files.items():
        image = cv2.imread(file_name)
        if image is None:
            print(f"Error: Image file {file_name} not found or could not be loaded.")
            continue
        resized_image = cv2.resize(image, (tile_size, tile_size))
        image_dict[key] = resized_image

    # Ensure all images were loaded and resized
    if len(image_dict) != len(image_files):
        print("Error: Not all images were loaded successfully.")
        return

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

if __name__ == "__main__":
    display_matrix(matrix)
