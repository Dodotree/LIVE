#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tkinter as tk

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


# In[7]:


import math

POINT_RADIUS = 2

class DataPoint:
    def __init__(self, x, y):
        # Save parameters.
        self.x = x
        self.y = y

    # Return the distance between this point and another one.
    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    # Draw the point on the canvas.
    def draw(self, canvas):
        canvas.create_oval(
            self.x - POINT_RADIUS, self.y - POINT_RADIUS,
            self.x + POINT_RADIUS, self.y + POINT_RADIUS,
            fill='green', outline='green')

    def __str__(self):
        return f'({self.x}, {self.y})'


# In[8]:


import math

class Cluster:
    def __init__(self, data_point):
        self.data_points = [ data_point ]
        self.connections = []

    # Draw the cluster's connections and data points.
    def draw(self, canvas):
        # Draw the connections.
        for connection in self.connections:
            canvas.create_line(
                connection[0].x, connection[0].y,
                connection[1].x, connection[1].y,
                fill='red')

        # Draw the data points.
        for data_point in self.data_points:
            data_point.draw(canvas)

    # Find the shortest distance between this cluster's
    # data points and another cluster's data points.
    # Return the best distance and the closest points
    # form this cluster and the other one.
    # P1

    def distance(self, other):
        best_distance = float('inf')
        best_my_point = None
        best_other_point = None

        for my_point in self.data_points:
            for other_point in other.data_points:
                dist = math.sqrt((my_point.x - other_point.x) ** 2 + (my_point.y - other_point.y) ** 2)
                if dist < best_distance:
                    best_distance = dist
                    best_my_point = my_point
                    best_other_point = other_point

        return best_distance, best_my_point, best_other_point

    # Merge with another cluster by adding its data points and connections.
    def consume_cluster(self, other, new_connection):
        self.data_points.extend(other.data_points)
        self.connections.extend(other.connections)
        self.connections.append(new_connection)

# In[9]:


import tkinter as tk
from tkinter import messagebox
import random

# Geometry constants.
WINDOW_WID = 500
WINDOW_HGT = 300
MARGIN = 5
CANVAS_WID = WINDOW_WID - 200
CANVAS_HGT = WINDOW_HGT - 2 * MARGIN

class App:
    # Create and manage the tkinter interface.
    def __init__(self):
        self.running = False
        self.data_points = []
        self.clusters = None

        # Make the main interface.
        self.window = tk.Tk()
        self.window.title('hierarchical_clustering_2d')
        self.window.protocol('WM_DELETE_WINDOW', self.kill_callback)
        self.window.geometry(f'{WINDOW_WID}x{WINDOW_HGT}')

        # Build the rest of the UI.
        self.build_ui()

        # Display the window.
        self.window.focus_force()
        self.window.mainloop()

    def build_ui(self):
        # Drawing canvas.
        self.canvas = tk.Canvas(self.window, bg='white',
            borderwidth=1, highlightthickness=0,
            width=CANVAS_WID, height=CANVAS_HGT)
        self.canvas.pack(side=tk.LEFT, padx=MARGIN, pady=MARGIN)
        self.canvas.bind('<Button-1>', self.left_click)

        # Right frame.
        right_frame = tk.Frame(self.window, pady=MARGIN)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Clusters.
        self.num_clusters_entry = make_field(
            right_frame, 11, '# Clusters:', 5, '2')

        # Delay (ms).
        self.delay_entry = make_field(
            right_frame, 11, 'Delay (ms):', 5, '500')

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

        # Run button.
        self.run_button = tk.Button(right_frame,
            text='Run', width=7, command=self.run, state=tk.DISABLED)
        self.run_button.pack(side=tk.TOP, pady=(20, 0))

        # Reset button.
        self.reset_button = tk.Button(right_frame,
            text='Reset', width=7, command=self.reset, state=tk.DISABLED)
        self.reset_button.pack(side=tk.TOP, pady=(MARGIN, 0))

        # Clear button.
        self.clear_button = tk.Button(right_frame,
            text='Clear', width=7, command=self.clear, state=tk.DISABLED)
        self.clear_button.pack(side=tk.TOP, pady=(MARGIN, 0))

    def left_click(self, event):
        self.data_points.append(DataPoint(event.x, event.y))
        self.set_button_states()
        self.redraw()

    def set_button_states(self):
        if len(self.data_points) > 0 and not self.running:
            self.reset_button['state'] = tk.NORMAL
            self.clear_button['state'] = tk.NORMAL
        else:
            self.reset_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED

        if len(self.data_points) > 0:
            self.run_button['state'] = tk.NORMAL
        else:
            self.run_button['state'] = tk.DISABLED

    def set_button_states(self):
        if len(self.data_points) > 0 and not self.running:
            self.reset_button['state'] = tk.NORMAL
            self.clear_button['state'] = tk.NORMAL
        else:
            self.reset_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED

        if len(self.data_points) > 0:
            self.run_button['state'] = tk.NORMAL
        else:
            self.run_button['state'] = tk.DISABLED

    # Stop running.
    def stop_running(self):
        self.running = False
        self.run_button.config(text='Run')
        self.set_button_states()

    # Start running.
    def start_running(self):
        if len(self.data_points) < 1:
            messagebox.showinfo('Data Points Error',
                'You must define at least one data point.')
            return

        # Get parameters.
        self.num_clusters = get_int(self.num_clusters_entry)
        if self.num_clusters < 1:
            messagebox.showinfo('# Clusters Error',
                'You must create at least one cluster.')
            return

        self.running = True
        self.run_button.config(text='Stop')
        self.set_button_states()

        # If we don't already have clusters, make some.
        if self.clusters is None:
            # Create the initial clusters containing one point each.
            self.clusters = [Cluster(data_point) for data_point in self.data_points]

        # Go!
        self.tick()

    def run(self):
        # See if we are currently running.
        if self.running:
            self.stop_running()
        else:
            self.start_running()

    # Perform one round of hierarchical clustering.

    # Merge two clusters.
    def tick(self):
        # If we have the desired number of clusters, stop.
        if len(self.clusters) <= self.num_clusters:
            # Stop running.
            self.stop_running()
            return

        # See which two clusters are closest together.
        best_distance = math.inf
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                dist, my_point, other_point = self.clusters[i].distance(self.clusters[j])
                if dist < best_distance:
                    best_distance = dist
                    best_i = i
                    best_j = j
                    best_my_point = my_point
                    best_other_point = other_point

        # Merge clusters i and j.
        new_connection = (best_my_point, best_other_point)
        self.clusters[best_i].consume_cluster(self.clusters[best_j], new_connection)
        del self.clusters[best_j]

        # Draw the points and clusters.
        self.redraw()

        # If we're still running, schedule another tick.
        if self.running:
            self.window.after(get_int(self.delay_entry), self.tick)


    # Reset the clusters so we can run again with the same points.
    def reset(self):
        self.running = False
        self.clusters = None
        self.redraw()
        self.set_button_states()

    # Destroy all DataPoints and graphics.
    def clear(self):
        self.running = False
        self.clusters = None
        self.data_points = []
        self.redraw()
        self.set_button_states()
        self.canvas.delete('all')

    def kill_callback(self):
        self.window.destroy()

    # Redraw everything.
    def redraw(self):
        self.canvas.delete('all')

        # See if we have any clusters.
        if self.clusters is not None:
            # Draw clusters.
            for cluster in self.clusters:
                cluster.draw(self.canvas)
        else:
            # Draw data points.
            for point in self.data_points:
                point.draw(self.canvas)

    def load_dataset_1(self):
        self.stop_running()
        self.canvas.delete('all')
        self.clusters = None
        self.data_points = [
            DataPoint(62, 80),
            DataPoint(82, 58),
            DataPoint(95, 91),
            DataPoint(111, 54),
            DataPoint(80, 82),
            DataPoint(136, 86),
            DataPoint(121, 108),
            DataPoint(106, 75),
            DataPoint(96, 105),
            DataPoint(67, 124),
            DataPoint(165, 217),
            DataPoint(166, 198),
            DataPoint(193, 219),
            DataPoint(225, 237),
            DataPoint(207, 248),
            DataPoint(171, 260),
            DataPoint(150, 234),
            DataPoint(184, 240),
            DataPoint(184, 264),
            DataPoint(176, 222),
            DataPoint(194, 199),
            DataPoint(212, 216),
            DataPoint(240, 98),
            DataPoint(215, 101),
            DataPoint(220, 129),
            DataPoint(223, 113),
            DataPoint(242, 122),
            DataPoint(253, 113),
            DataPoint(244, 85),
            DataPoint(219, 72),
            DataPoint(235, 144),
            DataPoint(266, 131),
            DataPoint(259, 92),
            DataPoint(205, 119),
            DataPoint(63, 100),
        ]
        self.redraw()
        self.set_button_states()

    def load_dataset_2(self):
        self.stop_running()
        self.canvas.delete('all')
        self.clusters = None
        self.data_points = [
            DataPoint(198, 69),
            DataPoint(215, 75),
            DataPoint(213, 99),
            DataPoint(220, 127),
            DataPoint(211, 149),
            DataPoint(63, 192),
            DataPoint(92, 208),
            DataPoint(164, 209),
            DataPoint(91, 68),
            DataPoint(54, 107),
            DataPoint(50, 134),
            DataPoint(136, 59),
            DataPoint(174, 58),
            DataPoint(212, 191),
            DataPoint(202, 170),
            DataPoint(192, 194),
            DataPoint(167, 192),
            DataPoint(143, 192),
            DataPoint(129, 209),
            DataPoint(142, 225),
            DataPoint(101, 228),
            DataPoint(99, 189),
            DataPoint(72, 220),
            DataPoint(45, 181),
            DataPoint(70, 179),
            DataPoint(55, 160),
            DataPoint(36, 160),
            DataPoint(36, 140),
            DataPoint(45, 150),
            DataPoint(42, 113),
            DataPoint(60, 68),
            DataPoint(59, 88),
            DataPoint(99, 56),
            DataPoint(82, 93),
            DataPoint(127, 36),
            DataPoint(151, 53),
            DataPoint(150, 20),
            DataPoint(124, 48),
            DataPoint(200, 48),
            DataPoint(180, 40),
            DataPoint(166, 35),
            DataPoint(224, 96),
            DataPoint(240, 136),
            DataPoint(238, 115),
            DataPoint(230, 114),
            DataPoint(223, 133),
            DataPoint(231, 158),
            DataPoint(216, 177),
            DataPoint(206, 176),
            DataPoint(183, 179),
            DataPoint(195, 212),
            DataPoint(138, 127),
            DataPoint(133, 114),
            DataPoint(155, 114),
            DataPoint(151, 131),
            DataPoint(145, 120),
            DataPoint(142, 142),
            DataPoint(131, 133),
            DataPoint(125, 123),
            DataPoint(124, 144),
        ]
        self.redraw()
        self.set_button_states()

    def load_dataset_3(self):
        self.stop_running()
        self.canvas.delete('all')
        self.clusters = None
        self.data_points = [
            DataPoint(100, 87),
            DataPoint(92, 62),
            DataPoint(74, 84),
            DataPoint(123, 75),
            DataPoint(140, 76),
            DataPoint(174, 76),
            DataPoint(202, 77),
            DataPoint(190, 60),
            DataPoint(155, 67),
            DataPoint(189, 83),
            DataPoint(218, 113),
            DataPoint(207, 97),
            DataPoint(233, 85),
            DataPoint(230, 100),
            DataPoint(193, 116),
            DataPoint(187, 128),
            DataPoint(179, 114),
            DataPoint(199, 123),
            DataPoint(173, 142),
            DataPoint(167, 133),
            DataPoint(167, 160),
            DataPoint(156, 161),
            DataPoint(157, 145),
            DataPoint(113, 172),
            DataPoint(135, 153),
            DataPoint(140, 169),
            DataPoint(126, 164),
            DataPoint(90, 188),
            DataPoint(103, 191),
            DataPoint(115, 187),
            DataPoint(129, 195),
            DataPoint(129, 176),
            DataPoint(103, 195),
            DataPoint(86, 221),
            DataPoint(69, 212),
            DataPoint(67, 228),
            DataPoint(83, 238),
            DataPoint(107, 212),
            DataPoint(106, 235),
            DataPoint(139, 259),
            DataPoint(124, 253),
            DataPoint(117, 253),
            DataPoint(125, 240),
            DataPoint(183, 253),
            DataPoint(207, 228),
            DataPoint(207, 231),
            DataPoint(209, 244),
            DataPoint(202, 240),
            DataPoint(199, 256),
            DataPoint(182, 238),
            DataPoint(169, 248),
            DataPoint(147, 241),
            DataPoint(151, 258),
            DataPoint(170, 260),
            DataPoint(64, 130),
            DataPoint(64, 143),
            DataPoint(50, 137),
            DataPoint(51, 123),
            DataPoint(48, 157),
            DataPoint(43, 152),
            DataPoint(59, 152),
            DataPoint(37, 135),
            DataPoint(218, 163),
            DataPoint(220, 169),
            DataPoint(235, 173),
            DataPoint(223, 152),
            DataPoint(248, 152),
            DataPoint(227, 164),
            DataPoint(247, 176),
            DataPoint(239, 155),
            DataPoint(239, 189),
            DataPoint(227, 179),
            DataPoint(211, 180),
            DataPoint(95, 76),
            DataPoint(114, 74),
            DataPoint(114, 74),
            DataPoint(114, 74),
            DataPoint(118, 57),
            DataPoint(145, 57),
        ]
        self.redraw()
        self.set_button_states()

    def load_dataset_4(self):
        self.stop_running()
        self.canvas.delete('all')
        self.clusters = None
        self.data_points = [
            DataPoint(139, 31),
            DataPoint(127, 60),
            DataPoint(137, 117),
            DataPoint(137, 160),
            DataPoint(147, 120),
            DataPoint(115, 96),
            DataPoint(141, 90),
            DataPoint(152, 60),
            DataPoint(156, 112),
            DataPoint(123, 74),
            DataPoint(68, 241),
            DataPoint(80, 228),
            DataPoint(115, 249),
            DataPoint(135, 240),
            DataPoint(155, 219),
            DataPoint(169, 242),
            DataPoint(193, 248),
            DataPoint(120, 219),
            DataPoint(155, 255),
            DataPoint(211, 229),
            DataPoint(190, 221),
            DataPoint(245, 232),
        ]
        self.redraw()
        self.set_button_states()


# In[10]:


App()


# In[ ]:




