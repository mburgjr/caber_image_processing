import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from image_sequence import ImageSequence

class Video():
    '''
    Class to represent a CABER video which can be broken into a sequence of images for study

    Inputs:
        file_path                   Path to video (String, Required)
        playback_fps                Playback FPS of the video provided (Int, Required)
        crop                        Flag of whether or not you want to crop the video (Boolean, Default: True)
        pixel_format                FFMPEG pixel format, must be one of ['yuvj420p', 'rgb24', 'uyvy422', 'bgra', 'yuv444p', 'yuv410p', 'monow', 'yuv420p'] (String, Default: 'yuv420p')
        folder_path                 Path to folder where image frames will be stored, must have no images in it (String, Default: './temp/{filename}')
        poly_degree                 Degree of polynomial approximation for either edge (Int, Default: 4)
        time_btwn_frames            Time between image frames / radius measurements (Float, Default: 1/playback_fps)
        lower_canny_threshold       Lower threshold for Canny filtering, decrease to increase edge count (Int, Range: [0, 255], Default: 25)
        upper_canny_threshold       Upper threshold for Canny filtering, increase to increase edge count (Int, Range: [0, 255], Default: 230)
        edge_benchmark              Light above this value should be considered an edge when searching the Canny filtered image (Int, Range: [0, 255], Default: 60)
        pct_considered              Vertical center percentage of the image considered for radius measurement, polynomial approximated across this domain (Float, Range: (0, 1], Default: 0.4)
        width                       Width of the image in terms of real world length, for scale (Float, Default: 1)
        graph_title                 Title to place on the matplotlib graph (String, Default: '')
        verbose                     Flag whether or not to print statements (Bool, Default: True)
    '''

    def __init__(self, file_path, playback_fps, folder_path=None, crop=True, pixel_format='yuv420p', poly_degree=4, time_btwn_frames=None, lower_canny_threshold=25, upper_canny_threshold=230, edge_benchmark=60, pct_considered=0.4, width=1, graph_title='', verbose=True):
        assert (os.path.isfile(file_path) == True), 'Given image path is not a file.'
        assert (type(playback_fps) == int), 'Playback FPS must be an integer.'
        assert (type(crop) == bool), 'Crop flag must be a boolean.'
        assert (pixel_format in ['yuvj420p', 'rgb24', 'uyvy422', 'bgra', 'yuv444p', 'yuv410p', 'monow', 'yuv420p']), 'Invalid pixel format provided.'
        assert (poly_degree >= 0 and poly_degree % 1 == 0), 'Polynomial degree must be positive integer.'
        assert (lower_canny_threshold >= 0 and lower_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (upper_canny_threshold >= 0 and upper_canny_threshold <= 255), 'All canny thresholds must be between 0 and 255.'
        assert (edge_benchmark >= 0 and edge_benchmark <= 255), 'Edge search benchmark must be between 0 and 255.'
        assert (pct_considered > 0 and pct_considered <= 1), 'Vertical percentage of the image considered must be between 0 and 1.'
        assert (width >= 0), 'Image width must be positive.'
        assert (type(graph_title) == str), 'Graph title must be a string.'
        assert (type(verbose) == bool), 'Verbose flag must be boolean.'

        # Add video to class
        self.file_path = os.path.abspath(file_path)
        self.video = cv.VideoCapture(self.file_path)

        # Add attributes to class
        self.crop = crop
        self.pixel_format = pixel_format
        self.lower_canny_threshold = lower_canny_threshold
        self.upper_canny_threshold = upper_canny_threshold
        self.edge_benchmark = edge_benchmark # Above this value is considered an edge (after Canny detector applied)
        self.pct_considered = pct_considered # Middle percentage of the middle to consider for approximation
        self.poly_degree = poly_degree
        self.width = width # Length width of the image, in this case 6mm (radius will be the same units)
        self.graph_title = graph_title
        self.verbose = verbose

        # Set default time before frames based on FPS
        if time_btwn_frames == None:
            time_btwn_frames = 1 / playback_fps
        self.time_btwn_frames = time_btwn_frames

        # User can crop video to relevant portion only
        if crop == True:
            # Get last frame from the video
            accessible, frame = self.video.read()
            while accessible:
                accessible, frame = self.video.read()
                if accessible:
                    last_frame = frame

            # Allow user to choose crop region
            roi = cv.selectROI(last_frame) # Returns as [y, x, width, height]
            cv.destroyWindow('ROI selector')

            # Save crop region params as [height, width, x, y]
            crop_params = [roi[2], roi[3], roi[0], roi[1]]
            self.crop_params = crop_params

        # Give option for custom folder path
        if type(folder_path) != str:
            folder_path = './temp/' + file_path.split('/')[-1]
            
            if not os.path.isdir('./temp'):
                os.mkdir('./temp')
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

            # Clear temp path if anything exists in there
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
                    os.remove(folder_path + '/' + filename)

        self.folder_path = os.path.abspath(folder_path)

        # Make sure folder contains no images
        for filename in os.listdir(folder_path):
            assert (not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif'))), 'Given folder path must be empty.'

        # Unpack video into sequence of images
        command = 'ffmpeg -i ' + self.file_path
        if self.crop == True:
            crop_filter = str(self.crop_params[0]) + ':' + str(self.crop_params[1]) + ':' + str(self.crop_params[2]) + ':' + str(self.crop_params[3])
            command += ' -filter:v "crop=' + crop_filter + '"'
        command += ' -pix_fmt ' + self.pixel_format + ' ' + self.folder_path + '/%d.tif'
        os.system(command)

        self.image_sequence = ImageSequence(self.folder_path, poly_degree=self.poly_degree, time_btwn_frames=self.time_btwn_frames, lower_canny_threshold=self.lower_canny_threshold, 
                                            upper_canny_threshold=self.upper_canny_threshold, edge_benchmark=self.edge_benchmark, pct_considered=self.pct_considered, width=self.width, 
                                            graph_title=graph_title, crop=False, verbose=self.verbose)

        # Copy over some attributes
        self.images = self.image_sequence.images
        self.radius = self.image_sequence.radius
        self.fit_results = self.image_sequence.fit_results
        self.fit_parameters = self.image_sequence.fit_parameters
        self.fit_init_cond = self.image_sequence.fit_init_cond


    def plot_radius(self, time_btwn_frames=None):
        # Use image sequence methods to plot radius data
        if time_btwn_frames == None: time_btwn_frames = self.time_btwn_frames
        return self.image_sequence.plot_radius(time_btwn_frames=time_btwn_frames)


    def fit(self, equation_function, parameter_ranges, equation_args=None, init_cond=None, time_btwn_frames=None, time_start=None, time_end=None, pct_data_considered=1.0, resize_with_arc_length=False, method='leastsq', ode=True, radius_index=0, log_error=False, parameter_guesses={}, vary_parameter={}, range_sections=30, max_guess_loops=None, vary_init_cond=True, vary_init_cond_pct=0.1):
        # Use image sequence methods to fit radius data to equation
        if time_btwn_frames == None: time_btwn_frames = self.time_btwn_frames
        self.image_sequence.fit(equation_function, parameter_ranges, equation_args=equation_args, init_cond=init_cond, time_btwn_frames=time_btwn_frames, time_start=time_start, time_end=time_end, pct_data_considered=pct_data_considered, 
                                resize_with_arc_length=resize_with_arc_length, method=method, ode=ode, radius_index=radius_index, log_error=log_error, parameter_guesses=parameter_guesses, vary_parameter=vary_parameter, range_sections=range_sections, 
                                max_guess_loops=max_guess_loops, vary_init_cond=vary_init_cond, vary_init_cond_pct=vary_init_cond_pct)
        self.fit_results = self.image_sequence.fit_results
        self.fit_parameters = self.image_sequence.fit_parameters
        self.fit_init_cond = self.image_sequence.fit_init_cond


    def plot_fit(self, time_btwn_frames=None, time_start=None, time_end=None, log=True):
        # Use image sequence methods to plot radius data alongside fit equation
        if time_btwn_frames == None: time_btwn_frames = self.time_btwn_frames
        self.image_sequence.plot_fit(time_btwn_frames=time_btwn_frames, time_start=time_start, time_end=time_end, log=log)


if __name__ == '__main__':
    # Try example video fit
    video_path = '../data/1wtp_2_7d7_6000fps_2.mp4'
    playback_fps = 30
    time_btwn_frames = 0.149*(10**(-3)) # in seconds
    crop_width = 6*(10**(-3)) # in meters
    vid = Video(video_path, playback_fps, poly_degree=4, pct_considered=0.4, time_btwn_frames=time_btwn_frames, width=crop_width, graph_title='Stock-PG0d4_2000fps_1.mp4')
    # vid.plot_radius() # Time in ms

    # Fit to Oldroyd-B model
    parameter_ranges = {
        'G': (1, 100),
        'gamma': (1*10**(-3), 100*10**(-3)),
        'eta_S': (0.001, 0.2),
        'lamb': (0.001, 0.2)
    }
    parameter_guesses = {
        'G': 50,
        'gamma': 55*10**(-3),
        'eta_S': 0.1,
        'lamb': 0.1
    }
    initial_condition = np.array([0.003, 67, 0])
    pct_of_data_to_use = 0.25
    vid.fit('oldroyd_b', parameter_ranges, parameter_guesses=parameter_guesses, init_cond=initial_condition, pct_data_considered=pct_of_data_to_use, time_start=0.04, time_end=0.115)
    print(vid.fit_parameters)
    vid.plot_fit(log=True)