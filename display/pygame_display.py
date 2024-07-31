import pygame
import sys
import numpy as np

# Initialize Pygame

clock = pygame.time.Clock()

# Define a matrix (example: 1 for image1, 2 for image2)
matrix_test = np.array(([
    [14, 14, 2, 4, 3, 1, 4, 2, 4],
    [14, 14, 2, 3, 1, 2, 1, 3, 3],
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
    [14, 14, 2, 4, 3, 1, 4, 2, 4],
    [14, 14, 1, 3, 1, 2, 1, 3, 3],
    [3, 4, 2, 5, 2, 4, 1, 2, 5],
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
        # pygame.init()
        self.rows, self.cols = matrix.shape
        self.cell_size = cell_size
        self.matrix = matrix

        # Define the window size
        self.window_width, self.window_height = cols * cell_size, rows * cell_size
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

        self.image_dict = {}
        for i in range(15):
            self.image_dict[i+1] = pygame.transform.scale(pygame.image.load(fr'./display/{i+1}.jpg'),(cell_size, cell_size))
        self.update_display(self.matrix)

    def update_display(self, new_matrix):
        self.window.fill((255, 255, 255))

        # Iterate through the matrix and blit images to the window
        for row in range(self.rows):
            for col in range(self.cols):
                image = self.image_dict.get(new_matrix[row][col])
                if image:
                    x, y = row * self.cell_size, col * self.cell_size
                    self.window.blit(image, (y, x))

        # Update the display
        pygame.display.flip()
        clock.tick(60)
    # the start and end positions that we are given will be the matrices positions.
    def animate_switch(self, start_pos, end_pos, matrix, steps=30):
        # Unpack start and end positions
        row, col = start_pos
        row2, col2 = end_pos

        # Retrieve the images
        white = self.image_dict.get(15)
        image1 = self.image_dict.get(matrix[row][col])
        image2 = self.image_dict.get(matrix[row2][col2])

        for step in range(steps):
            # Calculate intermediate positions
            inter_row = row * self.cell_size + (row2 - row) * self.cell_size * step / steps
            inter_col = col * self.cell_size + (col2 - col) * self.cell_size * step / steps
            inter_row2 = row2 * self.cell_size + (row - row2) * self.cell_size * step / steps
            inter_col2 = col2 * self.cell_size + (col - col2) * self.cell_size * step / steps
        
            # Ensure intermediate positions are integers
            inter_row = int(inter_row)
            inter_col = int(inter_col)
            inter_row2 = int(inter_row2)
            inter_col2 = int(inter_col2)

            # Draw intermediate positions
            self.window.blit(white, (inter_col, inter_row))
            self.window.blit(white, (inter_col2, inter_row2))
            self.window.blit(image1, (inter_col, inter_row))
            self.window.blit(image2, (inter_col2, inter_row2))

            pygame.display.flip()
            pygame.time.wait(10)  # Adjust the wait time for smoother/faster animation


if __name__ == "__main__":
    display = Display(matrix_test)
    animate = True
    # run the display. we can update the code and stuff beneaths
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if animate:

            display.animate_switch((1,1),(1,2), display.matrix)
            display.update_display(display.matrix)

       