import pygame
import sys
import numpy as np

# Initialize Pygame


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

# Define a matrix (example: 1 for image1, 2 for image2)
matrix_test = np.array(([
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
]))

matrix_test2 = np.array(([
    [14, 13, 2, 4, 3, 1, 4, 2, 4],
    [14, 14, 4, 3, 1, 2, 1, 3, 3],
    [3, 4, 1, 5, 2, 4, 1, 2, 5],
    [5, 5, 4, 5, 2, 5, 5, 4, 4],
    [4, 1, 2, 3, 1, 2, 3, 4, 2],
    [4, 1, 4, 4, 2, 4, 1, 3, 4],
    [2, 4, 3, 3, 5, 5, 4, 1, 2],
    [1, 2, 1, 1, 3, 3, 1, 4, 1],
    [4, 1, 3, 2, 1, 2, 1, 5, 2],
    [3, 2, 1, 2, 4, 2, 3, 2, 1]
]))


class Display():
    def __init__(self, matrix, rows = 10, cols = 9, cell_size = 64):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.matrix = matrix

        # Define the window size
        self.window_width, self.window_height = cols * cell_size, rows * cell_size
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

        self.image_dict = {}
        for i in range(14):
            self.image_dict[i+1] = pygame.transform.scale(pygame.image.load(fr'./display/{i+1}.png'),(cell_size, cell_size))

    def update_display(self, new_matrix):
        self.window.fill((255, 255, 255))

        # Iterate through the matrix and blit images to the window
        for row in range(self.rows):
            for col in range(self.cols):
                image = self.image_dict.get(new_matrix[row][col])
                if image:
                    x, y = col * self.cell_size, row * self.cell_size
                    self.window.blit(image, (x, y))

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    display = Display(matrix_test)
    display.update_display(matrix_test2)
    
    # run the display. we can update the code and stuff beneath
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

       