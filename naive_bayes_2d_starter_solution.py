#!/usr/bin/env python
# coding: utf-8

# # naive_bayes_2d
# Let the user draw points and then use naive Bayes to classify new points.

# In[151]:


import tkinter as tk
import numpy as np

# Get the text in an Entry widget and
# convert it to an int.

def get_int(entry):
    return int(entry.get())

# Make Label and Entry widgets for a field.
# Return the Entry widget.
def make_field(parent, label_width, label_text, entry_width, entry_default):
    frame = tk.Frame(parent)
    frame.pack(side=tk.TOP)

    label = tk.Label(frame, text=label_text, width=label_width, anchor=tk.W)
    label.pack(side=tk.LEFT)

    entry = tk.Entry(frame, width=entry_width, justify='right')
    entry.insert(tk.END, entry_default)
    entry.pack(side=tk.LEFT)

    return entry

# In[152]:

import math

class DataPoint:
    # Create a DataPoint at this spot.
    def __init__(self, x, y, name):
        self.name = name
        self.x = x
        self.y = y

    # Draw the data point.
    def create_oval(self, canvas, bg_color):
        radius = 8
        canvas.create_oval(
            self.x - radius, self.y - radius,
            self.x + radius, self.y + radius,
            fill=bg_color)
        canvas.create_text(self.x, self.y, text=self.name)

    # Use naive Bayes to classify the point.
    def naive_bayes(self, cluster_names, fractions, means, std_devs):
        DEBUG = False
        if DEBUG:
            print(cluster_names)
            print(fractions)
            print(means)   
            print(std_devs)

        properties = [self.x, self.y]

        best_prob = -1
        best_cluster_num = -1

        for cluster_num in range(len(cluster_names)):
            # Calculate the probability for this cluster.
            prob = fractions[cluster_num]
            for property_num in range(len(properties)):
                prob *= calculate_probability(
                    properties[property_num],
                    means[cluster_num][property_num],
                    std_devs[cluster_num][property_num])

            # Compare prob to best_prob and update if necessary
            if prob > best_prob:
                best_prob = prob
                best_cluster_num = cluster_num

        # Set the current DataPoint object's name property to the best cluster's name
        self.name = cluster_names[best_cluster_num]


# In[153]:


# Separate points into clusters.
# Return a cluster dictionary where dictionary[cluster_name]
# is a list of points assigned to this cluster.

def make_cluster_dictionary(data_points):
    cluster_dictionary = {}

    for point in data_points:
        # The 'name' attribute is the cluster name
        cluster_name = point.name
        data_point = point

        # If the cluster name is not in the dictionary, add an empty list
        if cluster_name not in cluster_dictionary:
            cluster_dictionary[cluster_name] = []

        # Append the data point to the appropriate list
        cluster_dictionary[cluster_name].append(data_point)

    return cluster_dictionary

# Calculate the mean and std dev for the properties in each cluster.
# Return lists holding:
#    cluster names
#    fraction of objects in this cluster
#    means
#    std devs
# In the lists, x values are in entry 0 and y values are in entry 1.
def summarize_points(num_points, cluster_dictionary):
    MIN_STD_DEV = 0.1

    cluster_names = []
    fractions = []
    means = []
    std_devs = []

    for cluster_name, data_points in cluster_dictionary.items():
        # Add the cluster's name to the list
        cluster_names.append(cluster_name)

        # Calculate and add the fraction of points in this cluster
        fractions.append(len(data_points) / num_points)

        # Calculate the means of the x and y coordinates
        cluster_means = [
            sum(data_point.x for data_point in data_points) / len(data_points),
            sum(data_point.y for data_point in data_points) / len(data_points),
        ]
        means.append(cluster_means)

        # Calculate the standard deviations of the x and y coordinates
        x_std_dev = np.std([data_point.x for data_point in data_points]) or MIN_STD_DEV
        y_std_dev = np.std([data_point.y for data_point in data_points]) or MIN_STD_DEV
        std_devs.append([x_std_dev, y_std_dev])

    return cluster_names, fractions, means, std_devs


# Calculate the Gaussian probability distribution function for x.
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2 )))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# In[154]:


import tkinter as tk

# The main App class.

# Geometry constants.
WINDOW_WID = 500
WINDOW_HGT = 300
MARGIN = 5
CANVAS_WID = WINDOW_WID - 200
CANVAS_HGT = WINDOW_HGT - 2 * MARGIN

class App:
    # Create and manage the tkinter interface.
    def __init__(self):
        self.network = None

        # Make the main interface.
        self.window = tk.Tk()
        self.window.title('naive_bayes_2d')
        self.window.protocol('WM_DELETE_WINDOW', self.kill_callback)
        self.window.geometry(f'{WINDOW_WID}x{WINDOW_HGT}')

        # Build the rest of the UI.
        self.build_ui()

        # Initially we have no data points.
        self.data_points = []

        # Display the window.
        self.window.focus_force()
        self.window.mainloop()

    # Build the tkinter user interface.
    def build_ui(self):
        # Make the drawing canvas.
        canvas_wid = CANVAS_WID
        canvas_hgt = CANVAS_HGT
        self.canvas = tk.Canvas(self.window, bg='white',
            borderwidth=0, highlightthickness=0, relief=tk.SUNKEN, width=canvas_wid, height=canvas_hgt)
        self.canvas.pack(side=tk.LEFT, padx=MARGIN, pady=MARGIN)
        self.canvas.bind('<Button-1>', self.click)

        # Right frame.
        right_frame = tk.Frame(self.window)
        right_frame.pack(side=tk.TOP, padx=MARGIN, pady=MARGIN)

        # Label and text box for a point's name.
        self.cluster_entry = make_field(right_frame, 10, 'Cluster:', 3, 'a')

        # Test data set buttons.
        button_frame = tk.Frame(right_frame, pady=MARGIN)
        button_frame.pack(side=tk.TOP)
        test1_button = tk.Button(button_frame,
            text='Dataset 1', width=8, command=self.load_dataset_1)
        test1_button.pack(side=tk.LEFT)
        test2_button = tk.Button(button_frame,
            text='Dataset 2', width=8, command=self.load_dataset_2)
        test2_button.pack(side=tk.LEFT, padx=(MARGIN, 0))

        # Test data set buttons.
        button_frame = tk.Frame(right_frame, pady=MARGIN)
        button_frame.pack(side=tk.TOP)
        test3_button = tk.Button(button_frame,
            text='Dataset 3', width=8, command=self.load_dataset_3)
        test3_button.pack(side=tk.LEFT)
        test4_button = tk.Button(button_frame,
            text='Dataset 4', width=8, command=self.load_dataset_4)
        test4_button.pack(side=tk.LEFT, padx=(MARGIN, 0))

        # Classify button.
        button_frame = tk.Frame(right_frame, pady=MARGIN)
        button_frame.pack(side=tk.TOP)
        classify_button = tk.Button(button_frame,
            text='Classify', width=8, command=self.classify)
        classify_button.pack(side=tk.LEFT)

        # Clear button.
        clear_button = tk.Button(right_frame, text='Clear', width=7, command=self.clear)
        clear_button.pack(side=tk.TOP, pady=10)

    # Clear existing points.
    def clear(self):
        self.data_points = []
        self.canvas.delete('all')

    # Save and draw a data point.
    def click(self, event):
        # Make the basic data point object.
        name = self.cluster_entry.get().strip()
        self.make_data_point(event.x, event.y, name)

    def make_data_point(self, x, y, name):
        data_point = DataPoint(x, y, name)

        # If it has no name, use naive Bayes to assign one.
        if name == '':
            # If there are no points defined yet, do nothing.
            if len(self.data_points) == 0: return

            # Give the point a cluster name.
            data_point.naive_bayes(self.cluster_names,
                self.fractions, self.means, self.std_devs)

            # Draw with a pink background.
            data_point.create_oval(self.canvas, 'pink')
        else:
            # Draw with a white background.
            data_point.create_oval(self.canvas, 'white')

            # Save this point to use later as training data.
            self.data_points.append(data_point)

    def kill_callback(self):
        self.window.destroy()

    def load_dataset_1(self):
        self.clear()
        self.data_points = [
            DataPoint(62, 80, 'a'),
            DataPoint(82, 58, 'a'),
            DataPoint(95, 91, 'a'),
            DataPoint(111, 54, 'a'),
            DataPoint(80, 82, 'a'),
            DataPoint(136, 86, 'a'),
            DataPoint(121, 108, 'a'),
            DataPoint(106, 75, 'a'),
            DataPoint(96, 105, 'a'),
            DataPoint(67, 124, 'a'),
            DataPoint(63, 100, 'a'),
            DataPoint(165, 217, 'c'),
            DataPoint(166, 198, 'c'),
            DataPoint(193, 219, 'c'),
            DataPoint(225, 237, 'c'),
            DataPoint(207, 248, 'c'),
            DataPoint(171, 260, 'c'),
            DataPoint(150, 234, 'c'),
            DataPoint(184, 240, 'c'),
            DataPoint(184, 264, 'c'),
            DataPoint(176, 222, 'c'),
            DataPoint(194, 199, 'c'),
            DataPoint(212, 216, 'c'),
            DataPoint(240, 98, 'b'),
            DataPoint(215, 101, 'b'),
            DataPoint(220, 129, 'b'),
            DataPoint(223, 113, 'b'),
            DataPoint(242, 122, 'b'),
            DataPoint(253, 113, 'b'),
            DataPoint(244, 85, 'b'),
            DataPoint(219, 72, 'b'),
            DataPoint(235, 144, 'b'),
            DataPoint(266, 131, 'b'),
            DataPoint(259, 92, 'b'),
            DataPoint(205, 119, 'b'),
        ]
        for point in self.data_points:
            point.create_oval(self.canvas, 'white')
        self.cluster_entry.delete(0, tk.END)
        self.cluster_entry.insert(tk.END, '')

    def load_dataset_2(self):
        self.clear()
        self.data_points = [
            DataPoint(198, 69, 'a'),
            DataPoint(215, 75, 'a'),
            DataPoint(213, 99, 'a'),
            DataPoint(220, 127, 'a'),
            DataPoint(211, 149, 'a'),
            DataPoint(63, 192, 'a'),
            DataPoint(92, 208, 'a'),
            DataPoint(164, 209, 'a'),
            DataPoint(91, 68, 'a'),
            DataPoint(54, 107, 'a'),
            DataPoint(50, 134, 'a'),
            DataPoint(136, 59, 'a'),
            DataPoint(174, 58, 'a'),
            DataPoint(212, 191, 'a'),
            DataPoint(202, 170, 'a'),
            DataPoint(192, 194, 'a'),
            DataPoint(167, 192, 'a'),
            DataPoint(143, 192, 'a'),
            DataPoint(129, 209, 'a'),
            DataPoint(142, 225, 'a'),
            DataPoint(101, 228, 'a'),
            DataPoint(99, 189, 'a'),
            DataPoint(72, 220, 'a'),
            DataPoint(45, 181, 'a'),
            DataPoint(70, 179, 'a'),
            DataPoint(55, 160, 'a'),
            DataPoint(36, 160, 'a'),
            DataPoint(36, 140, 'a'),
            DataPoint(45, 150, 'a'),
            DataPoint(42, 113, 'a'),
            DataPoint(60, 68, 'a'),
            DataPoint(59, 88, 'a'),
            DataPoint(99, 56, 'a'),
            DataPoint(82, 93, 'a'),
            DataPoint(127, 36, 'a'),
            DataPoint(151, 53, 'a'),
            DataPoint(150, 20, 'a'),
            DataPoint(124, 48, 'a'),
            DataPoint(200, 48, 'a'),
            DataPoint(180, 40, 'a'),
            DataPoint(166, 35, 'a'),
            DataPoint(224, 96, 'a'),
            DataPoint(240, 136, 'a'),
            DataPoint(238, 115, 'a'),
            DataPoint(230, 114, 'a'),
            DataPoint(223, 133, 'a'),
            DataPoint(231, 158, 'a'),
            DataPoint(216, 177, 'a'),
            DataPoint(206, 176, 'a'),
            DataPoint(183, 179, 'a'),
            DataPoint(195, 212, 'a'),
            DataPoint(138, 127, 'b'),
            DataPoint(133, 114, 'b'),
            DataPoint(155, 114, 'b'),
            DataPoint(151, 131, 'b'),
            DataPoint(145, 120, 'b'),
            DataPoint(142, 142, 'b'),
            DataPoint(131, 133, 'b'),
            DataPoint(125, 123, 'b'),
            DataPoint(124, 144, 'b'),
        ]
        for point in self.data_points:
            point.create_oval(self.canvas, 'white')
        self.cluster_entry.delete(0, tk.END)
        self.cluster_entry.insert(tk.END, '')

    def load_dataset_3(self):
        self.clear()
        self.data_points = [
            DataPoint(100, 87, 'a'),
            DataPoint(92, 62, 'a'),
            DataPoint(74, 84, 'a'),
            DataPoint(123, 75, 'a'),
            DataPoint(140, 76, 'a'),
            DataPoint(174, 76, 'a'),
            DataPoint(202, 77, 'a'),
            DataPoint(190, 60, 'a'),
            DataPoint(155, 67, 'a'),
            DataPoint(189, 83, 'a'),
            DataPoint(218, 113, 'a'),
            DataPoint(207, 97, 'a'),
            DataPoint(233, 85, 'a'),
            DataPoint(230, 100, 'a'),
            DataPoint(193, 116, 'a'),
            DataPoint(187, 128, 'a'),
            DataPoint(179, 114, 'a'),
            DataPoint(199, 123, 'a'),
            DataPoint(173, 142, 'a'),
            DataPoint(167, 133, 'a'),
            DataPoint(167, 160, 'a'),
            DataPoint(156, 161, 'a'),
            DataPoint(157, 145, 'a'),
            DataPoint(113, 172, 'a'),
            DataPoint(135, 153, 'a'),
            DataPoint(140, 169, 'a'),
            DataPoint(126, 164, 'a'),
            DataPoint(90, 188, 'a'),
            DataPoint(103, 191, 'a'),
            DataPoint(115, 187, 'a'),
            DataPoint(129, 195, 'a'),
            DataPoint(129, 176, 'a'),
            DataPoint(103, 195, 'a'),
            DataPoint(86, 221, 'a'),
            DataPoint(69, 212, 'a'),
            DataPoint(67, 228, 'a'),
            DataPoint(83, 238, 'a'),
            DataPoint(107, 212, 'a'),
            DataPoint(106, 235, 'a'),
            DataPoint(139, 259, 'a'),
            DataPoint(124, 253, 'a'),
            DataPoint(117, 253, 'a'),
            DataPoint(125, 240, 'a'),
            DataPoint(183, 253, 'a'),
            DataPoint(207, 228, 'a'),
            DataPoint(207, 231, 'a'),
            DataPoint(209, 244, 'a'),
            DataPoint(202, 240, 'a'),
            DataPoint(199, 256, 'a'),
            DataPoint(182, 238, 'a'),
            DataPoint(169, 248, 'a'),
            DataPoint(147, 241, 'a'),
            DataPoint(151, 258, 'a'),
            DataPoint(170, 260, 'a'),
            DataPoint(95, 76, 'a'),
            DataPoint(114, 74, 'a'),
            DataPoint(114, 74, 'a'),
            DataPoint(114, 74, 'a'),
            DataPoint(118, 57, 'a'),
            DataPoint(145, 57, 'a'),
            DataPoint(64, 130, 'b'),
            DataPoint(64, 143, 'b'),
            DataPoint(50, 137, 'b'),
            DataPoint(51, 123, 'b'),
            DataPoint(48, 157, 'b'),
            DataPoint(43, 152, 'b'),
            DataPoint(59, 152, 'b'),
            DataPoint(37, 135, 'b'),
            DataPoint(218, 163, 'c'),
            DataPoint(220, 169, 'c'),
            DataPoint(235, 173, 'c'),
            DataPoint(223, 152, 'c'),
            DataPoint(248, 152, 'c'),
            DataPoint(227, 164, 'c'),
            DataPoint(247, 176, 'c'),
            DataPoint(239, 155, 'c'),
            DataPoint(239, 189, 'c'),
            DataPoint(227, 179, 'c'),
            DataPoint(211, 180, 'c'),
        ]
        for point in self.data_points:
            point.create_oval(self.canvas, 'white')
        self.cluster_entry.delete(0, tk.END)
        self.cluster_entry.insert(tk.END, '')

    def load_dataset_4(self):
        self.clear()
        self.data_points = [
            DataPoint(139, 31, 'a'),
            DataPoint(127, 60, 'a'),
            DataPoint(137, 117, 'a'),
            DataPoint(137, 160, 'a'),
            DataPoint(147, 120, 'a'),
            DataPoint(115, 96, 'a'),
            DataPoint(141, 90, 'a'),
            DataPoint(152, 60, 'a'),
            DataPoint(156, 112, 'a'),
            DataPoint(123, 74, 'a'),
            DataPoint(68, 241, 'b'),
            DataPoint(80, 228, 'b'),
            DataPoint(115, 249, 'b'),
            DataPoint(135, 240, 'b'),
            DataPoint(155, 219, 'b'),
            DataPoint(169, 242, 'b'),
            DataPoint(193, 248, 'b'),
            DataPoint(120, 219, 'b'),
            DataPoint(155, 255, 'b'),
            DataPoint(211, 229, 'b'),
            DataPoint(190, 221, 'b'),
            DataPoint(245, 232, 'b'),
        ]
        for point in self.data_points:
            point.create_oval(self.canvas, 'white')
        self.cluster_entry.delete(0, tk.END)
        self.cluster_entry.insert(tk.END, '')

    def classify(self):
        # Define some rectangle colors.
        colors = [
            'red', 'green', 'blue', 'lightgreen', 'pink',
            'lightblue', 'cyan', 'magenta', 'yellow', 'orange',
            'gray'
        ]

        # Prepare for classification.
        # Separate the clusters.
        self.cluster_dictionary = make_cluster_dictionary(self.data_points)

        # Calculate mean and std dev for the x and y properties.
        self.cluster_names, self.fractions, self.means, self.std_devs = \
            summarize_points(len(self.data_points), self.cluster_dictionary)

        # Redraw the points.
        self.canvas.delete('all')
        for data_point in self.data_points:
            data_point.create_oval(self.canvas, 'white')

        # Display the clusters graphically.
        for i in range(len(self.cluster_names)):
            cx = self.means[i][0]
            cy = self.means[i][1]
            x_radius = self.std_devs[i][0]
            y_radius = self.std_devs[i][1]
            for devs in range(1, 4):
                self.canvas.create_rectangle(
                    cx - devs * x_radius,
                    cy - devs * y_radius,
                    cx + devs * x_radius,
                    cy + devs * y_radius,
                    outline=colors[i],
                    dash=(2))


# In[ ]:


App()


# In[ ]:




