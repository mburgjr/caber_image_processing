import numpy as np
import math
import matplotlib.pyplot as plt

class Bezier():

    '''
    Class that can be used to change the size of a dataset of points by fitting a Bezier curve between each of the points and sampling
    Reference: https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2

    Inputs:
        points      List of datapoints to be resized (Required, List/Array)
    '''

    def __init__(self, points):
        # Save points to object
        if type(points) == list:
            points = np.array(points)
        self.points = points
        self.data_size = self.points.shape[0]

        # Create coefficients for points
        num_curves = len(points) - 1
        assert num_curves > 0, 'Error: not enough data found.'

        # Build coefficents matrix
        C = 4*np.identity(num_curves)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[num_curves - 1, num_curves - 1] = 7
        C[num_curves - 1, num_curves - 2] = 2

        # Build points vector
        P = [2*(2*points[i] + points[i + 1]) for i in range(num_curves)]
        P[0] = points[0] + 2*points[1]
        P[num_curves - 1] = 8*points[num_curves - 1] + points[num_curves]

        # Solve system, find A & B
        A = np.linalg.solve(C, P)
        B = [0]*num_curves
        for i in range(num_curves - 1):
            B[i] = 2*points[i + 1] - A[i + 1]
        B[num_curves - 1] = (A[num_curves - 1] + points[num_curves]) / 2

        # Save coefficients
        self.A = A
        self.B = B

        # Create curves
        curves = [self.curve_function(points[i], A[i], B[i], points[i + 1]) for i in range(len(points) - 1)]
        self.curves = curves


    def curve_function(self, a, b, c, d):
        # Return function object which evaluates to point at t given 4 control points
        # Where t is of the set (0, 1)
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


    def resize_dataset(self, n):
        # Change the size of the dataset points to length n
        # By using sequence of Bezier curves to create / select points in between
        new_dataset = []
        spacing = self.data_size/(n-1)

        # Loop size of new dataset
        for i in range(n-1):
            # Select curve
            t = spacing*i
            curve_index = int(t // 1)
            if (curve_index >= len(self.curves)):
                curve_index = len(self.curves) - 1

            # Choose R along the curve
            R = min(max(0, t - curve_index), 1) # Fit to range (0,1)            
            point = self.curves[curve_index](R)

            # Add point from R on curve
            new_dataset.append(point)

        # Add engpoint to make it length n
        new_dataset.append(self.points[-1])

        # Return numpy array
        new_dataset = np.array(new_dataset)
        return new_dataset


    def resize_dataset_via_arc_length(self, n):
        # Change the size of the dataset points to length n
        # By selecting equidistantly along linearly approximated arc length
        points = self.points
        assert len(points[0]) == 2, 'Must be 2D to use arc length selection, higher dimensional not implemented'

        segment_lengths = [] # Where element i is linear length from point[i] to point[i+1]
        slopes = [] # Where element i is linear slope from point[i] to point[i+1]

        # Get distance between points and linear slopes
        for i in range(1, len(self.points)):
            delta_x = points[i][0] - points[i-1][0]
            delta_y = points[i][1] - points[i-1][1]
            segment_length = ((delta_x)**2 + (delta_y)**2)**0.5
            segment_lengths.append(segment_length)
            slopes.append(delta_y/delta_x)
        
        total_arc_length = sum(segment_lengths)

        # Initialize variables
        resized_points = []
        last_point = 0
        leftover_length = 0
        length_btwn_points = total_arc_length / (n-1)
        resized_points.append(points[0])

        # Loop output size
        for _ in range(n-1):
            desired = length_btwn_points - leftover_length
            
            # Loop points until have the desired distance between points
            length = 0
            for i in range(last_point, len(points)):
                if i == len(segment_lengths):
                    resized_points.append(points[-1])
                    break
            
                length += segment_lengths[i]
                if length >= desired:
                    # Add to resized dataset
                    last_point = i
                    leftover_length = length - desired
                    slope = slopes[i]
                    x = points[i][0] + (segment_lengths[i] - leftover_length)*math.cos(math.atan(slope))
                    y = points[i][1] + (segment_lengths[i] - leftover_length)*math.sin(math.atan(slope))
                    resized_points.append([x, y])
                    break

        return resized_points


# Export function
def resize_dataset(points, n, arc_length_method=False):
    bezier = Bezier(points)
    if arc_length_method == True:
        return bezier.resize_dataset_via_arc_length(n)
    else:
        return bezier.resize_dataset(n)