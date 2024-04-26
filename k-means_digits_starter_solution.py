#!/usr/bin/env python
# coding: utf-8

# # K-means Digits
# Load data and find clusters. Then let the user draw digits and use the clusters to identify them.

# In[ ]:


import math

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

    # Return the distance between this point and another one.
    def distance(self, other):
        total = 0
        for i in range(len(self.properties)):
            total += abs(self.properties[i] - other.properties[i])
        return math.sqrt(total)

    # Assign this DataPoint to the seed that is closest.
    def assign_seed(self, seeds):
        # Find the closest seed.
        self.seed = None
        best_distance = math.inf
        for seed in seeds:
            # If the distance to this seed is shorter than
            # the best distance so far, save the new seed.
            test_distance = self.distance(seed)
            if best_distance > test_distance:
                best_distance = test_distance
                self.seed = seed

    # Reposition this seed given its currently assigned data points.
    # Return the distance moved.
    def reposition_seed(self, training_points):
        # Save a copy of the seed's point (so we have its location).
        old_point = DataPoint(self.data_string)
        old_point.properties = self.properties.copy()

        # Reset this seed's properties.
        self.reset_properties()

        # Loop through the data points assigned to this seed
        # and add each point's properties to this seed's.
        num_assigned = 0
        for training_point in training_points:
            if training_point.seed == self:
                num_assigned += 1
                self.add_properties(training_point)

        # See if any points are assigned to this seed.
        if num_assigned == 0:
            # No points assigned. Don't move the seed.
            # Restore the old property values.
            self.properties = old_point.properties
            return 0

        # Average the propery values.
        self.divide_properties(num_assigned)

        # Return the distance moved.
        return self.distance(old_point)

    # Reset all properties to 0.
    def reset_properties(self):
        for i in range(len(self.properties)):
            self.properties[i] = 0

    # Add the other DataPoint's properties to this one's properties.
    def add_properties(self, other):
        for i in range(len(self.properties)):
            self.properties[i] += other.properties[i]

    # Divide each property value by num (to get an average).
    def divide_properties(self, num):
        for i in range(len(self.properties)):
            self.properties[i] /= num

    # Count the names of the training points assigned to
    # this seed and give it the name with the most votes.
    def assign_name(self, training_points):
        # Tally the votes.
        votes = {}
        for training_point in training_points:
            if training_point.seed == self:
                if training_point.name in votes:
                    votes[training_point.name] += 1
                else:
                    votes[training_point.name] = 1

        # See which name had the most votes.
        if len(votes) == 0:
            self.name = '?'
        else:
            self.name = max(votes, key=lambda x:votes[x])


# In[ ]:


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
        self.window.title('k-means digits')
        self.window.protocol('WM_DELETE_WINDOW', self.kill_callback)
        self.window.geometry(f'{WINDOW_WID}x{WINDOW_HGT}')

        # Build the UI.
        self.build_ui()

        # Load the data.
        self.load_data()

        # Test K values between 3 and 20.
        # (We want 10 results so there's probably
        # no point using fewer than 10 clusters.)
        self.test_ks(3, 20)

        # Display the final results.
        self.success_rate_value.set(f'K = {self.k}, Success Rate = {self.success_rate}%')
        print(f'Final: K = {self.k}, Success Rate = {self.success_rate}%')
        print('Seeds:')
        for seed in self.seeds:
            print(f'    {seed.name}')

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

    # See which seed is closest to the polyline.
    def evaluate_polyline(self):
        # Convert the polyline into a DataPoint.
        data_point = self.polyline_to_data_point()

        # Assign the DataPoint to a seed.
        data_point.assign_seed(self.seeds)

        # Display the result.
        self.user_result_value.set(data_point.seed.name)
        print(f'Digit: {data_point.seed.name}')

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

    #########################
    ### K-Means Functions ###
    #########################

    # Test different values for K.
    # Save the best results in self.k, self.success_rate, and self.seeds.
    # Test different values for K.
    # Save the best results in self.k, self.success_rate, and self.seeds.
    def test_ks(self, min_k, max_k):
        self.k = 0
        self.success_rate = 0
        self.seeds = []

        # Test K values between min_k and max_k.
        for k in range(min_k, max_k + 1):
            test_success_rate, test_seeds = self.test_data(k)

            # If this is an improvement, update self.k.
            if self.success_rate < test_success_rate:
                self.k = k
                self.success_rate = test_success_rate
                self.seeds = test_seeds


    # Test each of the data points with this value for K.
    # Save the best total distance, success rate, distance, and seeds list.
    def test_data(self, k):
        # Start with no solution.
        best_success_rate = 0
        best_seeds = []

        # Repeat several times to find a good set of clusters for this K.
        NUM_TRIALS = 20  # Maybe use a bigger number like 100.
        for trial in range(NUM_TRIALS):
            # Randomize the data.
            random.shuffle(self.data_points)

            # Divide the data into training and test points.
            num_training = int(len(self.data_points) * 0.75)
            training_points = self.data_points[:num_training]
            test_points = self.data_points[num_training:]

            # Find seeds.
            MAX_ITERATIONS = 1000
            STOP_DISTANCE = 1
            test_success_rate, test_seeds = self.find_clusters(
            k, training_points, test_points, MAX_ITERATIONS, STOP_DISTANCE)

            # See if this is an improvement.
            if best_success_rate < test_success_rate:
                # print(f'Improvement: K = {k}, Success Rate = {test_success_rate:.2f}%')
                best_success_rate = test_success_rate
                best_seeds = test_seeds

        # Print the results.
        print(f'K = {k}, Success Rate = {best_success_rate:.2f}%')
        return best_success_rate, best_seeds


    # Assign points to their nearest seeds.
    def assign_points_to_seeds(self, data_points, seeds):
        for data_point in data_points:
            data_point.assign_seed(seeds)

    # Reposition the seeds.
    # Return the largest distance that any seed moves.
    def reposition_seeds(self, data_points, seeds):
        max_move = 0
        for seed in seeds:
            distance_moved = seed.reposition_seed(data_points)
            if max_move < distance_moved:
                max_move = distance_moved
        return max_move

    # Calculate the success rate percentage.
    def calculate_success_rate(self, seeds, test_points):
        num_correct = 0
        for test_point in test_points:
            test_point.assign_seed(seeds)
            if test_point.seed.name == test_point.name:
                num_correct += 1
        return int(100 * num_correct / len(test_points))

    # Inputs:
    #     k:                The number of seeds to use (K).
    #     training_points:  A list of DataPoint objects to use when making the centroids.
    #     test_points:      A list of DataPoint objects to use to test success rate.
    #     max_iterations:   The maximum number of iterations we will perform.
    #     stop_distance:    When the change in total distance is less than this, we stop looping.
    # Returns:
    #     The success rate percentage.
    #     The list of seeds.
    def find_clusters(self, k, training_points, test_points,
                      max_iterations=1000, stop_distance=1):

        # Make k initial seeds.
        seeds = []
        for seed in random.sample(training_points, k):
            # Make a copy of this data point so
            # we don't mess up the original.
            seeds.append(DataPoint(seed.data_string))

        # Repeat until things stabilize.
        for iteration in range(max_iterations):
            # Assign points to their nearest seeds.
            self.assign_points_to_seeds(training_points, seeds)
            # Move the seeds to their centroids.
            if self.reposition_seeds(training_points, seeds) < stop_distance:
                break

        # Assign likely names to seeds.
        for seed in seeds:
            # Assign the seed's name.
            seed.assign_name(training_points)

        # Calculate the success rate percentage.
        success_rate = self.calculate_success_rate(seeds, test_points) 

        # Return the results.
        return success_rate, seeds

# In[ ]:


App()


# In[ ]:




