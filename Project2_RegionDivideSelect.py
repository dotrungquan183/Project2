import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from operator import add
from functools import reduce
import cv2

# Load the image
img = cv2.imread('icon/Test_Image.png', 0)

# Define functions for quadtree processing
def split4(img):
    half_split = np.array_split(img, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)

def calculate_mean(img):
    return np.mean(img, axis=(0, 1))

def checkEqual(myList):
    first = myList[0]
    return all((x == first).all() for x in myList)

class QuadTree:
    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = (img.shape[0], img.shape[1])
        self.final = True

        if not checkEqual(img):
            split_img = split4(img)
            self.final = False
            self.north_west = QuadTree().insert(split_img[0], level + 1)
            self.north_east = QuadTree().insert(split_img[1], level + 1)
            self.south_west = QuadTree().insert(split_img[2], level + 1)
            self.south_east = QuadTree().insert(split_img[3], level + 1)

        return self

    def get_image(self, level):
        if self.final or self.level == level:
            return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))

        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level))

# Initialize the QuadTree
quadtree = QuadTree().insert(img)

# Setup Tkinter GUI
root = tk.Tk()
root.title("QuadTree Image Segmentation")

# Frame for original image and segmented images
frame_original = ttk.LabelFrame(root, text="Original Image")
frame_original.grid(row=0, column=0, padx=10, pady=10)

frame_segmented = ttk.LabelFrame(root, text="Segmented Image")
frame_segmented.grid(row=0, column=1, padx=10, pady=10)

frame_histogram = ttk.LabelFrame(root, text="Histogram")
frame_histogram.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Display Original Image
fig1, ax1 = plt.subplots()
ax1.imshow(img, cmap="gray")
ax1.set_title("Original Image")
canvas1 = FigureCanvasTkAgg(fig1, master=frame_original)
canvas1.draw()
canvas1.get_tk_widget().pack()

# Display Segmented Image for a specific level (e.g., level 8)
fig2, ax2 = plt.subplots()
segmented_img = quadtree.get_image(10)
ax2.imshow(segmented_img, cmap="gray")
ax2.set_title("Segmented Image (Level 8)")
canvas2 = FigureCanvasTkAgg(fig2, master=frame_segmented)
canvas2.draw()
canvas2.get_tk_widget().pack()

# Display Histogram
fig3, ax3 = plt.subplots()
ax3.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
ax3.set_title("Histogram")
canvas3 = FigureCanvasTkAgg(fig3, master=frame_histogram)
canvas3.draw()
canvas3.get_tk_widget().pack()

# Run the Tkinter loop
root.mainloop()