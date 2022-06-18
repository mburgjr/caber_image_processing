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

See ./examples for example scripts
