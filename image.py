import os

from numpy.core import numeric
from numba import jit
import numpy as np
from scipy import optimize
import cv2 as cv
from matplotlib import pyplot as plt

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit(nopython=True)
def extract_edge_points(edge_image, benchmark, pct_considered):
    # Given Canny edge image, find location of edge pixels
    left_edge_points = []
    right_edge_points = []

    # Get list of the rows to look at, only considering a vertical middle percent of the image
    start_row = int(edge_image.shape[0]*(1 - pct_considered)/2)
    end_row = int(edge_image.shape[0]*(0.5 + pct_considered/2))
    rows_considered = range(start_row, end_row, 1)

    columns = range(edge_image.shape[1])

    # Loop rows considered
    found_left_last_row = False
    found_right_last_row = False
    for i in rows_considered:
        # Set flag to see if either edge was found on this row yet
        found_left = False
        found_right = False

        # Search through columns
        for j_left in columns:
            # Get reverse direction indedx for the right edge search
            j_right = len(columns) - j_left - 1

            # If met, don't add points
            if abs(j_right - j_left) == 0:
                break

            # Check if have found an edge from the left
            if edge_image[i][j_left] > benchmark and (not found_left):
                # Make sure the previous row connects
                if found_left_last_row == False or left_edge_points.count((i-1, j_left)) + left_edge_points.count((i-1, j_left-1)) + left_edge_points.count((i-1, j_left+1)) + left_edge_points.count((i-1, j_left-2)) + left_edge_points.count((i-1, j_left+2)) > 0:
                    found_left = True
                    left_edge_points.append((i, j_left))
            elif edge_image[i][j_left] > benchmark and left_edge_points.count((i, j_left-1)) > 0:
                # Continuation on the same row
                found_left = True
                left_edge_points.append((i, j_left))

            # Check if have found an edge from the right
            if edge_image[i][j_right] > benchmark and (not found_right):
                # Make sure the previous row connects
                if found_right_last_row == False or right_edge_points.count((i-1, j_right)) + right_edge_points.count((i-1, j_right-1)) + right_edge_points.count((i-1, j_right+1)) + right_edge_points.count((i-1, j_right-2)) + right_edge_points.count((i-1, j_right+2)) > 0:
                    found_right = True
                    right_edge_points.append((i, j_right))
            elif edge_image[i][j_right] > benchmark and right_edge_points.count((i, j_right-1)) > 0:
                # Continuation on the same row
                found_right = True
                right_edge_points.append((i, j_right))

            # If about to cross, don't check any further points on the row
            if abs(j_right - j_left) == 1:
                break

        # Save whether or not edges were found on this row
        found_left_last_row = found_left
        found_right_last_row = found_right

    return left_edge_points, right_edge_points


@jit(nopython=True)
def check_broken(image, benchmark, pct_considered, pixel_threshold=5):
    # Check if the liquid has snapped in the center
    fully_across = 0

    # Loop rows
    for i in range(image.shape[0]):
        # Search through columns
        for j in range(image.shape[1]):
            if image[i][j] < benchmark:
                fully_across = 0
                break

            if j == image.shape[1] - 1:
                fully_across += 1
                break

        if fully_across >= pixel_threshold:
            return True

    return False


def fit_polynomial(data, degree):
    # Fit polynomial to data points
    x = np.array([ point[0] for point in data ])
    y = np.array([ point[1] for point in data ])

    a = np.zeros(shape=(x.shape[0], degree + 1))
    const = np.ones_like(x)
    a[:, 0] = const
    a[:, 1] = x
    if degree > 1:
        for n in range(2, degree + 1):
            a[:, n] = x**n

    p = np.linalg.lstsq(a, y, rcond=-1)[0]

    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


def eval_polynomial(poly, x):
    # Evaluate polynomial at x
    result = []
    for i in range(len(poly)):
        coeff = poly[i]
        result.append(coeff*x**(len(poly) - i - 1))
    result = sum(result)
    return result


class Image():
    '''
    Class to represent a single CABER image

    Inputs:
        file_path                   Path to image (String, Required)
        poly_degree                 Degree of polynomial approximation for either edge (Int, Default: 4)
        lower_canny_threshold       Lower threshold for Canny filtering, decrease to increase edge count (Int, Range: [0, 255], Default: 25)
        upper_canny_threshold       Upper threshold for Canny filtering, increase to increase edge count (Int, Range: [0, 255], Default: 230)
        edge_benchmark              Light above this value should be considered white, to be used in edge location (Int, Range: [0, 255], Default: 60)
        pct_considered              Vertical center percentage of the image considered for radius measurement, polynomial approximated across this domain (Float, Range: (0, 1], Default: 0.4)
        width                       Width of the image in terms of real world length, for scale (Float, Default: 1)
        crop                        Flag whether or not to crop the provided image (bool, Default: True)
        verbose                     Flag whether or not to print statements (Bool, Default: True)
    '''

    def __init__(self, file_path, poly_degree=4, lower_canny_threshold=25, upper_canny_threshold=230, edge_benchmark=60, pct_considered=0.4, width=1, crop=True, verbose=True):
        assert (os.path.isfile(file_path) == True), 'Given image path is not a file.'
        assert (poly_degree >= 0 and poly_degree % 1 == 0), 'Polynomial degree must be positive integer.'
        assert (lower_canny_threshold >= 0 and lower_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (upper_canny_threshold >= 0 and upper_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (edge_benchmark >= 0 and edge_benchmark <= 255), 'Edge search benchmark must be between 0 and 255.'
        assert (pct_considered > 0 and pct_considered <= 1), 'Vertical percentage of the image considered must be between 0 and 1.'
        assert (width >= 0), 'Image width must be positive.'
        assert (type(crop) == bool or (type(crop) == list and len(crop) == 4)), 'Crop flag must be boolean or list of parameters.'
        assert (type(verbose) == bool), 'Verbose flag must be boolean.'

        # Add image to class
        self.file_path = os.path.abspath(file_path)
        image = cv.imread(self.file_path, 0)
        norm = np.zeros(image.shape)
        self.image = cv.normalize(image,  norm, 0, 255, cv.NORM_MINMAX)
        self.crop_params = None

        if crop == True:
            # Allow user to choose crop region
            roi = cv.selectROI(self.image) # Returns as [height, width, x, y]
            cv.destroyWindow('ROI selector')

            # Save crop region params as [height, width, x, y]
            crop_params = [roi[2], roi[3], roi[0], roi[1]]
            self.crop_params = crop_params

        elif type(crop) == list:
            # Set parameters based on input
            self.crop_params = crop

        # Crop the image
        if self.crop_params != None and self.crop_params != False:
            # Resize image matrix
            self.image = self.image[self.crop_params[3]:self.crop_params[3] + self.crop_params[1], self.crop_params[2]:self.crop_params[2] + self.crop_params[0]]

        # Define class attributes
        self.lower_canny_threshold = lower_canny_threshold
        self.upper_canny_threshold = upper_canny_threshold
        self.edge_benchmark = edge_benchmark # Above this value is considered an edge (after Canny detector applied)
        self.pct_considered = pct_considered # Middle percentage of the middle to consider for approximation
        self.poly_degree = poly_degree
        self.width = width # Length width of the image, in this case 6mm (radius will be the same units)
        self.verbose = verbose

        # Get edges
        self.canny_edges_image = cv.Canny(self.image, self.lower_canny_threshold, self.upper_canny_threshold)

        # Get polynomial edge approximations for either side
        self.left_edge_points, self.right_edge_points = self.get_edge_points()
        self.left_poly = self.get_poly(self.left_edge_points)
        self.right_poly = self.get_poly(self.right_edge_points)
        self.poly = self.right_poly - self.left_poly
        self.poly_image = self.get_poly_image()

        # Check if the liquid is broken and radius is zero
        if check_broken(self.image, self.edge_benchmark, self.pct_considered):
            self.radius = 0
        else:
            # Calculate radius
            self.radius = self.get_radius()

    def get_edge_points(self):
        # Extract left and right edge points
        left_edge_points, right_edge_points = extract_edge_points(self.canny_edges_image, self.edge_benchmark, self.pct_considered)
        return left_edge_points, right_edge_points

    def get_poly(self, data):
        # Get poly for given data points and degree
        poly = fit_polynomial(data, self.poly_degree)
        return poly

    def eval_poly(self, poly, x):
        # Evaluate given polynomial at x
        y = eval_polynomial(poly, x)
        return y

    def get_poly_image(self):
        # Generate image with polynomials mapped
        poly_image = np.zeros(self.image.shape)
        for x in range(self.image.shape[0]):
            left_val = self.eval_poly(self.left_poly, x)
            right_val = self.eval_poly(self.right_poly, x)
            
            # If in range, add to image
            if 0 <= left_val < self.image.shape[1]:
                poly_image[x][int(left_val)] = 1
            if 0 <= right_val < self.image.shape[1]:
                poly_image[x][int(right_val)] = 1

        return poly_image

    def get_radius(self):
        # Get start and end of clipped radius region
        start = int((0.5 - self.pct_considered/2)*self.image.shape[0])
        end = int((0.5 + self.pct_considered/2)*self.image.shape[0])

        # Define distance between poly's at given row via function
        def r(x):
            return self.eval_poly(self.poly, x)

        # Get min value over interval
        min_val = r(optimize.minimize(r, x0=(start + end)/2).x[0])
        min_val = max(0, min_val)

        # Scale to length
        min_R = min_val*((self.width / 2) / self.image.shape[1])
        return min_R

    def show_image(self):
        # Display original image
        plt.subplot(121), plt.imshow(self.image, cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def show_edge_image(self):
        # Display original image and Canny filtered side by side
        plt.subplot(121), plt.imshow(self.image, cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.canny_edges_image, cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def show_edge_data_image(self):
        # Display original image and edge data extracted from Canny filter side by side
        edge_data_image = np.zeros(self.image.shape)
        for point in self.left_edge_points:
            edge_data_image[point[0]][point[1]] = 1
        for point in self.right_edge_points:
            edge_data_image[point[0]][point[1]] = 1

        plt.subplot(121), plt.imshow(self.image, cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edge_data_image, cmap = 'gray')
        plt.title('Edge Data Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def show_poly_image(self):
        # Display original image and edge polynomials side by side
        plt.subplot(121), plt.imshow(self.image, cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.poly_image, cmap = 'gray')
        plt.title('Polynomials Image'), plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    # Try a sample image
    image_path = './data/sample_frames/0240.tif'
    crop_width = 6*(10**(-3)) # in meters
    im = Image(image_path, poly_degree=4, width=crop_width, crop=False)

    # View edge fitting and polynomial approximations
    im.show_edge_image()
    im.show_edge_data_image()
    im.show_poly_image()