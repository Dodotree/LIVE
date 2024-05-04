#!/usr/bin/env python
# coding: utf-8

# # naive_bayes_digits
# Load data and use naive Bayes to find clusters. Then let the user draw digits and use the clusters to identify them.

# In[56]:


import math
import numpy as np


class DataPoint:
    # The data_string parameter is a string holding the digit, 0s, and 1s
    # in the format '6: 011110110000100000111111110001110001010001001111'
    def __init__(self, data_string):
        self.data_string = data_string
        self.seed = None

        fields = data_string.split(' ')
        self.name = fields[0][0]

        self.properties = []
        for ch in fields[1].strip():
            self.properties.append(int(ch))

    # Use naive Bayes to classify the point.
    def naive_bayes(self, cluster_names, fractions, means, std_devs):
        DEBUG = False
        if DEBUG:
            print(cluster_names)
            print(fractions)
            print(means)
            print(std_devs)

        best_prob = -1
        best_cluster_num = -1
        for cluster_num in range(len(cluster_names)):
            # Calculate the probability for this cluster.
            prob = fractions[cluster_num]
            for property_num in range(len(self.properties)):
                prob *= calculate_probability(
                    self.properties[property_num],
                    means[cluster_num][property_num],
                    std_devs[cluster_num][property_num])
            if DEBUG:
                print(f'{cluster_names[cluster_num]} probability: {prob}')

            # Update the best probability.
            if prob > best_prob:
                best_prob = prob
                best_cluster_num = cluster_num
        if DEBUG:
            print()

        # See which cluster had the best probability.
        self.name = cluster_names[best_cluster_num]


# In[57]:


# Naive Bayes functions.
import math

# Separate points into clusters.
# Return a cluster dictionary where dictionary[cluster_name]
# is a list of points assigned to this cluster.
def make_cluster_dictionary(data_points):
    cluster_dictionary = {}
    for data_point in data_points:
        cluster_name = data_point.name
        if cluster_name not in cluster_dictionary:
            cluster_dictionary[cluster_name] = []
        cluster_dictionary[cluster_name].append(data_point)
    return cluster_dictionary

# Calculate the mean and std dev for the properties in each cluster.
# Return lists holding:
#    cluster names
#    fraction of objects in this cluster
#    means
#    std devs
# The means and std_devs lists contain lists of values for each cluster.
def summarize_points(num_points, cluster_dictionary):
    MIN_STD_DEV = 0.1

    cluster_names = []  # One name per cluster
    fractions = []      # Fraction of objects in each cluster
    means = []          # For each cluster, a list of means for each property
    std_devs = []       # For each cluster, a list of std devs for each property
    for cluster_name, data_points in cluster_dictionary.items():
        cluster_names.append(cluster_name)
        fractions.append(len(data_points) / num_points)

        # Calculate the means and std devs for the
        # properties of the points in this cluster.
        cluster_means = []
        cluster_std_devs = []

        for property_num in range(len(data_points[0].properties)):
            # Mean
            property_values = [data_point.properties[property_num] for data_point in data_points]
            mean = sum(property_values) / len(property_values)
            cluster_means.append(mean)

            # Std dev
            std_dev = np.std(property_values)
            if std_dev < MIN_STD_DEV:
                std_dev = MIN_STD_DEV
            cluster_std_devs.append(std_dev)


        std_devs.append(cluster_std_devs)
        means.append(cluster_means)

    return cluster_names, fractions, means, std_devs

# Calculate the Gaussian probability distribution function for x.
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean)**2 / (2 * stdev**2 )))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# In[58]:


import tkinter as tk
import random

# Geometry constants.
NUM_ROWS = 8
NUM_COLS = 6
CELL_WID = 20
CELL_HGT = CELL_WID
MARGIN = 5
WINDOW_WID = NUM_COLS * CELL_WID + 100
WINDOW_HGT = NUM_ROWS * CELL_HGT + 40

class App:
    # Create and manage the tkinter interface.
    def __init__(self):
        self.network = None

        # Make the main interface.
        self.window = tk.Tk()
        self.window.title('naive_bayes_digits')
        self.window.protocol('WM_DELETE_WINDOW', self.kill_callback)
        self.window.geometry(f'{WINDOW_WID}x{WINDOW_HGT}')

        # Build the UI.
        self.build_ui()

        # Load the data.
        self.load_data()

        # Train for naive Bayes classification.
        self.classify()

        # Calculate the success rate.
        self.calculate_success_rate(self.data_points)

        # Display the final results.
        self.success_rate_value.set(f'Success Rate = {self.success_rate}%')
        print(f'Final: Success Rate = {self.success_rate}%')

        # Initially we have nothing to draw.
        self.polyline = None
        self.points = []

        # Display the window.
        self.window.focus_force()
        self.window.mainloop()

    # Load the data and find good clusters.
    def load_data(self):
        # Load the DataPoints.
        with open('digit_data.txt', 'r') as f:
            lines = f.readlines()
            self.data_points = []
            for line in lines:
                self.data_points.append(DataPoint(line))

    def redraw(self):
        # Remove old polyline.
        self.canvas.delete(self.polyline)
        self.polyline = None

        # Draw current points.
        if len(self.points) > 1:
            self.polyline = self.canvas.create_line(self.points, fill='black')

    def build_ui(self):
        # Make the drawing canvas.
        canvas_wid = NUM_COLS * CELL_WID + 1
        canvas_hgt = NUM_ROWS * CELL_HGT + 1
        self.canvas = tk.Canvas(self.window, bg='white',
            borderwidth=0, highlightthickness=0, relief=tk.SUNKEN, width=canvas_wid, height=canvas_hgt)
        self.canvas.place(x=MARGIN, y=MARGIN)
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<ButtonRelease-1>', self.end_draw)

        # Make grid lines.
        for r in range(NUM_ROWS + 1):
            self.canvas.create_line(0, r * CELL_HGT, canvas_wid, r * CELL_HGT, fill='lime')
        for c in range(NUM_COLS + 1):
            self.canvas.create_line(c * CELL_WID, 0, c * CELL_WID, canvas_hgt, fill='lime')

        # Make a label to display success percentage.
        self.success_rate_value = tk.StringVar()
        self.success_rate_label = tk.Label(self.window, font=('Calibri 10 normal'), textvariable=self.success_rate_value)
        self.success_rate_label.place(x=MARGIN, y=canvas_hgt + 2 * MARGIN)

        # Make a big label to display results from the user drawing.
        self.user_result_value = tk.StringVar()
        self.user_result_label = tk.Label(self.window, font=('Calibri 100 normal'), textvariable=self.user_result_value)
        self.user_result_label.place(x=canvas_wid + 2 * MARGIN, y=MARGIN)

    def start_draw(self, event):
        # Clear any previous result.
        self.user_result_value.set('')

        # Remove any previous drawing.
        self.points = []
        self.redraw()

        self.canvas.bind('<B1-Motion>', self.save_point)

    def end_draw(self, event):
        self.canvas.unbind('<B1-Motion>')

        # Evaluate the polyline.
        self.evaluate_polyline()

    def save_point(self, event):
        self.points.append((event.x, event.y))
        self.redraw()

    def kill_callback(self):
        self.window.destroy()

    # See which cluster is most likely.
    def evaluate_polyline(self):
        # Convert the polyline into a DataPoint.
        data_point = self.polyline_to_data_point()

        # Give the point a cluster name.
        data_point.naive_bayes(self.cluster_names,
            self.fractions, self.means, self.std_devs)

        # Display the result.
        self.user_result_value.set(data_point.name)
        print(f'Digit: {data_point.name}')

    # Convert the polyline into a DataPoint.
    def polyline_to_data_point(self):
        # Convert the points into the cells that were touched.
        touched = self.get_touched()

        # Convert the touched cells to a string.
        touched_string = self.touched_to_string(touched)

        # Compose the DataPoint data string.
        data_string = f'?: {touched_string}'

        # Create the DataPoint.
        return DataPoint(data_string)

    # Convert the points into the cells that were touched.
    def get_touched(self):
        # Make a touched array holding 0s.
        touched = []
        for r in range(NUM_ROWS):
            touched.append([0 for i in range(NUM_COLS)])

        # Mark the touched cells.
        for point in self.points:
            r = int(point[1] / CELL_HGT)
            c = int(point[0] / CELL_WID)
            if r >= 0 and r < NUM_ROWS and c >= 0 and c < NUM_COLS:
                touched[r][c] = 1

        # Return the touched list.
        return touched

    # Return a string holding the touch values.
    def touched_to_string(self, touched):
        result = ''
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                result += str(touched[r][c])
        return result

    # Prepare for classification.
    def classify(self):
        # Separate the clusters.
        self.cluster_dictionary = make_cluster_dictionary(self.data_points)

        # Calculate mean and std dev for the x and y properties.
        self.cluster_names, self.fractions, self.means, self.std_devs = \
            summarize_points(len(self.data_points), self.cluster_dictionary)


    def calculate_success_rate(self, data_points):
        num_successes = 0

        for data_point in data_points:
            original_name = data_point.name
            data_point.naive_bayes(self.cluster_names, self.fractions, self.means, self.std_devs)
            if data_point.name == original_name:
                num_successes += 1

        self.success_rate = (num_successes / len(data_points)) * 100


# In[61]:


App()


# In[ ]:




