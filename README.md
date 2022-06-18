# CaBER Image Processing

Python library to process CaBER images. Given image(s) or video, edges are located to approximate the minimum radius of the liquid filament. Radius can then be plotted over time and approximated to an equation form.

# At a glance...

Capillary breakup rheometer (CaBER) measurements characterize the behavior of a fluid in extensional flow by stretching it in the vertical direction. As it is stretched, the radius gets smaller over time. The decreasing evolution of this radius can be evaluated to determine properties of the fluid. 

This repository was constructed to systematically process CaBER video/images in order to expedite data processing. Using an edge detection algorithm, pictures of the CaBER setup can be translated to radius measurements by finding the minimum distance across the fluid. Radius values can then be found for all frames to build a radius vs. time dataset.

The dataset of radius values can fit to an equation to gain insight on the fluid in question. Some equations will fit data better than others, it's all dependent on the type of fluid. A few standard models are provided as part of the dataset and can be easily used for fitting. Custom equations can also be provided following a standard input formatting which is better detailed below. Once the fit is complete, the radius time evolution is approximated and can be used in further calculations.


#
![alt text](https://github.mit.edu/raw/mburgjr/CABER-Image-Processing/master/caber.png?token=AAAELUR7EDEJSDSKXAW2IA3BPMKKU)


# Classes:

- **Image** - for processing singular images
- **ImageSequence** - for processing a set of images in chronological order
- **Video** - for processing an entire video frame-by-frame
- **Equation** - to represent an equation to which radius data can be fit


# Image:

**file_path** *(str, Required)* Path to image file.

**poly_degree** *(int, Default: 4)* Degree of polynomial approximation for fluid edges.

**lower_canny_threshold** *(int, Range: [0, 255], Default: 25)* Lower threshold for Canny filtering, decrease to increase edge count.

**upper_canny_threshold** *(int, Range: [0, 255], Default: 230)* Upper threshold for Canny filtering, increase to increase edge count.

**edge_benchmark** *(int, Range: [0, 255], Default: 60)* Light above this value should be considered white, to be used in edge location.

**pct_considered** *(float, Range: (0, 1], Default: 0.4)* Vertical center percentage of the image considered for radius measurement, polynomial approximated across this domain.

**width** *(float, Default: 1)* Real world width of the cropped image in meters, for scaling radius and fitting.

**crop** *(bool, Default: True)* Flag whether or not to crop the provided image. Can also be provided as a list of crop parameters of the form [height, width, x, y] where (x, y) is the location of the top-left corner of the crop in rows then columns.

**verbose** *(bool, Default: True)* Flag whether or not to print statements during processing.

**get_edge_points()** Returns a list of edge points on the original cropped image in units of pixels ordered row then column.

**get_poly(data)** Provided a list of datapoints of the form (x, y) (where x is row number and y is column number), return a list of polynomial coefficients according to the specified order of approximation.

**eval_poly(poly, x)** Provided a list of polynomial coefficients and some input value x, return the polynomial evaluated at x.

**get_poly_image()** Generate and return a matrix with polynomial approximations for both edges highlighted.

**get_radius()** Return the minimum radius of the fluid in the image in meters by taking the minimum between edge polynomials and scaling using the width parameter.

**show_image()** Display a picture of the raw cropped image.

**show_edge_image()** Display the raw cropped image with Canny edge filter applied and edges highlighted in white.

**show_edge_data_image()** Display the raw cropped image with edge data filter highlighted. This data is used to fit edge polynomials.

**show_poly_image()** Display polynomial edgelines that were found using datapoints highlighted in the show_edge_data_image display. The radius of the image is the minimum of the distance between these polynomials.


# ImageSequence:


# Video:


# Equation:
