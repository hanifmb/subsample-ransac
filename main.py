import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

def distance_to_line(point, slope, intercept):
    # Unpack the point coordinates
    x, y = point
    
    # Calculate the distance from the point to the line
    distance = abs(slope * x - y + intercept) / math.sqrt(slope ** 2 + 1)
    
    return distance

def sequential_ransac_multi_line_detection(data, threshold, min_points, max_iterations, num_of_lines_to_detect):

    best_lines = []
    color = [0] * len(data)
    remaining_data = data

    for i in range(num_of_lines_to_detect):
        best_line = ransac_line_detection(remaining_data, 0.1, 2, 5000)

        slope, intercept = best_line
        inlier_indices = []
        for j, (x, y) in enumerate(remaining_data):
            if distance_to_line((x, y), slope, intercept) < threshold:
                inlier_indices.append(j)

        remaining_data = [remaining_data[j] for j in range(len(remaining_data)) if j not in inlier_indices]

        best_lines.append(best_line)
    
    print(best_lines)
    
    color_code = 1
    for best_line in best_lines:
        slope, intercept = best_line
        for idx, (x, y) in enumerate(data):
            if distance_to_line((x, y), slope, intercept) < threshold:
                color[idx] = color_code 

        color_code += 1

    return color
    
def ransac_line_detection(data, threshold, min_points, max_iterations):

    best_line_model = None
    best_num_inliers = 0
    for i in range(max_iterations):
        # randomly select a subset of data points
        sample = random.sample(data, min_points)
        
        # fit a line to the subset of data points
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        if x2 - x1 == 0: # if the two sampled points have the same x-coordinate, skip this iteration
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # count the number of inliers (data points that are within the threshold distance of the line)
        num_inliers = 0
        for x, y in data:
            if distance_to_line((x, y), slope, intercept) < threshold:
                num_inliers += 1
        
        # update the best line model if this model has more inliers
        if num_inliers > best_num_inliers:
            best_line_model = (slope, intercept)
            best_num_inliers = num_inliers
    
    return best_line_model

def polar_to_cartesian(polar_points):

    cartesian_points = []
    for point in points:
        degrees = point[0]
        radius = point[1]

        radians = math.radians(degrees)
        x_cartesian = radius * math.cos(radians)
        y_cartesian = radius * math.sin(radians)

        cartesian_points.append((x_cartesian, y_cartesian))

    return cartesian_points

def load_points_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points


def onselect(eclick, erelease):
    # Get the coordinates of the selected rectangle
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    # Get the indices of the points within the selected rectangle
    ind = np.where((x_coords >= min(x1,x2)) & (x_coords <= max(x1,x2)) & (y_coords >= min(y1,y2)) & (y_coords <= max(y1,y2)))[0]
    
    # Print the indices of the selected points
    print('Selected points:', ind)

if __name__ == "__main__":

    points = load_points_file("3tabla1a.txt")
    cartesian_points = polar_to_cartesian(points)

    # best_line, color = ransac_line_detection(cartesian_points, 1.5, 2, 5000)
    color = sequential_ransac_multi_line_detection(cartesian_points, 2, 2, 5000, 4)

    # cartesian_points.append((0,0))
    # color = ['blue'] * len(cartesian_points)
    # color[-1] = 'red'

    global x_coords 
    x_coords = [p[0] for p in cartesian_points]
    global y_coords 
    y_coords = [p[1] for p in cartesian_points]

    # plt.xlim(0, 1000)
    # plt.ylim(-500, 500)

    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords)
    plt.scatter(x_coords, y_coords, c = color)

    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], 
                                       minspanx=5, minspany=5, spancoords='data')

    plt.show()
