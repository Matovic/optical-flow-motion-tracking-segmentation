# Optical flow, motion tracking, segmentation, stereo vision
Erik Matovič   
Methods used: 

## Assignment
### Data - pedestrian / road

Dataset : https://www.kaggle.com/datasets/smeschke/pedestrian-dataset?resource=download
### Sparse optical flow

Visualize trajectories of moving objects.

Optional task: Identify each object using a bounding box and count them.

Use following functions: cv::goodFeaturesToTrack, cv::calcOpticalFlowPyrLK

### Dense optical flow

Identify moving objects in video and draw green rectangle around them.

Use downsampled video for this task if necessary for easier processing.

Use following functions: cv::calcOpticalFlowFarneback

[OpenCV's tutorial on how to optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)

Motion tracking Datasets

Feel free to experiment with multiple videos for motion tracking. Use the following link for additional datasets - https://motchallenge.net/data/MOT15/

### Segmentation using background subtraction

Use background substraction methods to properly segment the moving objects from their background. Use one of the videos with static camera.

Use the following approaches:

    Accumulated weighted image

    Mixture of Gaussian (MOG2)


### Grab Cut segmentation

Propose a simple method to segment a rough estimate of lateral ventricle segmentation using morphological processing and thresholding.

[Link, 5 x PNG, 137 KB](https://drive.google.com/file/d/1hnQ_PHx0LhMNCMlpwCFhXVx4fFl9j_Aq/view) 

Use OpenCV's graph cut method to refine segmentation boundary.

cv::grabCut

Input has to be BGR (3 channel)

Values for the mask parameter:

GC_BGD = 0 - an obvious background pixels

GC_FGD = 1 - an obvious foreground (object) pixel

GC_PR_BGD = 2 - a possible background pixel

GC_PR_FGD = 3  - a possible foreground pixel

An example of GrabCut algorithm: link (note: This example uses a defined rectangle for grabcut segmentation. In our case we want to use the mask option instead)


## Usage
To run Jupyter Notebook, you need OpenCV and matplotlib. You can install them using pip:  
```bash
pip install opencv-python matplotlib
```

[OpenCV documentation](https://docs.opencv.org/4.7.0/)

## Solution
