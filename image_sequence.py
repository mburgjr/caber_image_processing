import os
import numpy as np
import time as Time
from multiprocessing import Pool
from matplotlib import pyplot as plt

import cv2 as cv

from image import Image
from equation import Equation
from bezier import resize_dataset

class ImageSequence():
    '''
    Class to represent a sequence of CABER images showing the progression of a filament over time

    Inputs:
        file_path                   Path to image (String, Required)
        poly_degree                 Degree of polynomial approximation for either edge (Int, Default: 4)
        time_btwn_frames            Time between image frames / radius measurements (Float, Default: 1)
        lower_canny_threshold       Lower threshold for Canny filtering, decrease to increase edge count (Int, Range: [0, 255], Default: 25)
        upper_canny_threshold       Upper threshold for Canny filtering, increase to increase edge count (Int, Range: [0, 255], Default: 230)
        edge_benchmark              Light above this value should be considered an edge when searching the Canny filtered image (Int, Range: [0, 255], Default: 60)
        pct_considered              Vertical center percentage of the image considered for radius measurement, polynomial approximated across this domain (Float, Range: (0, 1], Default: 0.4)
        width                       Width of the image in terms of real world length, for scale (Float, Default: 1)
        graph_title                 Title to place on the matplotlib graph (String, Default: '')
        crop                        Flag whether or not to crop the provided image (bool, Default: True)
        verbose                     Flag whether or not to print statements (Bool, Default: True)
    '''

    def __init__(self, folder_path, poly_degree=4, time_btwn_frames=1, lower_canny_threshold=25, upper_canny_threshold=230, edge_benchmark=60, pct_considered=0.4, width=1, graph_title='', crop=True, verbose=True):
        assert (os.path.isdir(folder_path) == True), 'Given image path is not a file.'
        assert (poly_degree >= 0 and poly_degree % 1 == 0), 'Polynomial degree must be positive integer.'
        assert (lower_canny_threshold >= 0 and lower_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (upper_canny_threshold >= 0 and upper_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (edge_benchmark >= 0 and edge_benchmark <= 255), 'Edge search benchmark must be between 0 and 255.'
        assert (pct_considered > 0 and pct_considered <= 1), 'Vertical percentage of the image considered must be between 0 and 1.'
        assert (width > 0), 'Image width must be positive and non-zero.'
        assert (type(graph_title) == str), 'Graph title must be a string.'
        assert (type(crop) == bool), 'Crop flag must be boolean.'
        assert (type(verbose) == bool), 'Verbose flag must be boolean.'

        # Add attributes to class
        self.folder_path = os.path.abspath(folder_path)
        self.time_btwn_frames = time_btwn_frames # Default to just integer frames
        self.lower_canny_threshold = lower_canny_threshold
        self.upper_canny_threshold = upper_canny_threshold
        self.edge_benchmark = edge_benchmark # Above this value is considered an edge (after Canny detector applied)
        self.pct_considered = pct_considered # Middle percentage of the middle to consider for approximation
        self.poly_degree = poly_degree
        self.width = width # Length width of the image, in this case 6mm (radius will be the same units)
        self.graph_title = graph_title
        self.crop_params = False
        self.verbose = verbose
        self.equation = None
        self.fit_results = None
        self.fit_parameters = {}
        self.fit_init_cond = np.array([])

        # Get crop params
        if crop == True:
            # Find last image
            for filename in sorted(os.listdir(self.folder_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
                    last_filename = filename

            # Load the image
            file_path = os.path.abspath(self.folder_path + '/' + last_filename)
            image = cv.imread(file_path, 0)
            norm = np.zeros(image.shape)
            image = cv.normalize(image,  norm, 0, 255, cv.NORM_MINMAX)

            # Allow user to choose crop region
            roi = cv.selectROI(image) # Returns as [y, x, width, height]
            cv.destroyWindow('ROI selector')

            # Save crop region params as [height, width, x, y]
            crop_params = [roi[2], roi[3], roi[0], roi[1]]
            self.crop_params = crop_params

        # Load images
        self.images = self.load_images()

        # Get radii
        self.radius = [im.radius for im in self.images]


    def load_image(self, filename):
        # Make sure file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
            file_path = self.folder_path + '/' + filename

            # Create image object, sped up using numba
            image = Image(file_path, poly_degree=self.poly_degree, lower_canny_threshold=self.lower_canny_threshold, upper_canny_threshold=self.upper_canny_threshold, 
                            edge_benchmark=self.edge_benchmark, pct_considered=self.pct_considered, width=self.width, crop=self.crop_params, verbose=self.verbose)
            return image


    def load_images(self):
        # Load and process images in parallel
        pool = Pool()
        images = pool.map(self.load_image, os.listdir(self.folder_path))
        images = [image for image in images if image != None]

        # Sort them
        self.images = sorted(images, key=lambda x: int(os.path.basename(x.file_path).replace('.tif', '')), reverse=False)
        return self.images


    def plot_radius(self, time_btwn_frames=None, log=False):
        # Get radius and time in correct units
        radius = np.array([image.radius for image in self.images])
        if time_btwn_frames == None: time_btwn_frames = self.time_btwn_frames
        time = time_btwn_frames*np.array(range(len(self.images)))

        # Plot radius versus time on log or linear scale
        if log:
            plt.semilogy(time, radius)
        else:
            plt.scatter(time, radius)
        plt.xlabel('Time (s)')
        plt.ylabel('Radius (m)')
        if self.graph_title != '':
            plt.title(self.graph_title)
        plt.show()


    def fit(self, equation_function, parameter_ranges, equation_args=None, init_cond=None, time_btwn_frames=None, time_start=None, time_end=None, pct_data_considered=1.0, resize_with_arc_length=False, method='leastsq', ode=True, radius_index=0, log_error=False, parameter_guesses={}, vary_parameter={}, range_sections=30, max_guess_loops=None, vary_init_cond=True, vary_init_cond_pct=0.2):
        # Fit radius data from images to differential equation
        assert (callable(equation_function) == True or type(equation_function) == str), 'Equation function must be a function.'

        # Make range bounds
        if time_start == None:
            time_start = 0
        if time_end == None:
            time_end = time_btwn_frames*len(self.images)

        # Create dataset of points
        points = []
        for i in range(len(self.images)):
            time_i = time_btwn_frames*i
            if time_i >= time_start and time_i <= time_end:
                radius_i = self.images[i].radius
                points.append([time_i, radius_i])

        # Downsize dataset to speed up fitting
        points = resize_dataset(points, int(len(points)*pct_data_considered), arc_length_method=resize_with_arc_length)
        time = [point[0] for point in points]
        radius = [point[1] for point in points]

        # Convert to arrays
        time = np.array(time)
        radius = np.array(radius)

        # Make equation object and fit
        self.equation = Equation(equation_function, parameter_ranges, equation_args=equation_args, method=method, ode=ode, radius_index=radius_index, log_error=log_error, verbose=self.verbose)
        self.fit_results = self.equation.fit(time, radius, init_cond=init_cond, parameter_guesses=parameter_guesses, vary_parameter=vary_parameter, range_sections=range_sections, max_guess_loops=max_guess_loops, vary_init_cond=vary_init_cond, vary_init_cond_pct=vary_init_cond_pct)
        
        # Unpack fit parameters and initial condition
        param_dict = self.fit_results.params.valuesdict()
        self.fit_init_cond = init_cond
        for key, value in param_dict.items():
            if type(self.fit_init_cond) != None and key.count('init_cond') > 0:
                i = int(key[9:])
                self.fit_init_cond[i] = value
            else:
                self.fit_parameters[key] = value


    def plot_fit(self, time_btwn_frames=None, time_start=None, time_end=None, pct_data_considered=1.0, log=True):
        # Plot radius data versus the equation fit to said data
        assert (self.equation != None and self.fit_results != None), 'Please run fit first.'

        # Make range bounds
        if time_start == None:
            time_start = 0
        if time_end == None:
            time_end = time_btwn_frames*len(self.images)

        # Create radius list for time range
        time = []
        radius = []
        for i in range(len(self.images)):
            time_i = time_btwn_frames*i + time_start
            if time_i >= time_start and time_i <= time_end:
                time.append(time_i)
                radius.append(self.images[i].radius)

        # Convert to arrays
        time = np.array(time)
        radius = np.array(radius)

        # Graph
        self.equation.plot_fit(time=time, radius=radius, params=self.fit_results.params, log=log)


if __name__ == '__main__':
    # Try a local example sequence of images
    print('Loading images...')
    start_time = Time.time()
    folder_path = './data/sample_frames'
    crop_width = 6*(10**(-3)) # meters
    image_seq = ImageSequence(folder_path, width=crop_width, crop=False)
    end_time = Time.time()
    seconds_elapsed = end_time - start_time
    print('Loaded ' + str(len(image_seq.images)) + ' images in ' + str(seconds_elapsed) + ' seconds.')
    time_btwn_frames = 0.149 # milliseconds
    image_seq.plot_radius(time_btwn_frames) # Time in ms


    

    '''
    points = []
    for i in range(len(image_seq.images)):
        time_i = time_btwn_frames*i
        radius_i = image_seq.images[i].radius
        points.append([time_i, radius_i])

    print(len(points))

    # Downsize dataset to speed up fitting
    points = resize_dataset(points, int(len(points)*0.05), arc_length_method=False)
    time = [point[0] for point in points]
    radius = [point[1] for point in points]

    
    plt.scatter(time, radius, color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Radius (m)')
    plt.show()
    '''