# Optical flow, motion tracking, segmentation, stereo vision
Erik Matovič   
Methods used: 

## Assignment
### Exploratory Data Analysis

For this experiment, we used annotated videos of crossing pedestrians in three scenarios: 
- a pedestrian is crossing a four-way until he reaches starting point,
- a pedestrian crossing at the crosswalk,
- a pedestrian walking and running in the night.

In the end, we used classical approaches of motion tracking utilizing sparse and dense optical flow. Thus labels were not needed because, for this experiment, we have not used a deep learning approach.

Dataset: https://www.kaggle.com/datasets/smeschke/pedestrian-dataset?resource=download


We have utilized OpenCV objects for working with video capture, and for saving outcomes as videos, we have used video writer:
```python3
def get_cap_out(video_path:str, out_root:str='..', start_idx:int=15) -> Tuple[cv2.VideoCapture,
                                                                              cv2.VideoWriter]:
    """
    Read video capture and make video writer.
    :param video_path:  path of the input 
    :param out_root:    path of the output folder
    :param start_idx:   index for the name of the output video 
    returns: cv2.VideoCapture, cv2.VideoWriter 
    """
    # load video
    cap = cv2.VideoCapture(video_path)

    # convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # make video writer
    out = cv2.VideoWriter(out_root + video_path[start_idx:-4] + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    return cap, out
```

### Sparse optical flow
Visualize trajectories of moving objects.

Use following functions: cv::goodFeaturesToTrack, cv::calcOpticalFlowPyrLK

Sparse optical flow with only 1 Shi-Tomasi Corner Detection and computation via Lucas-Kanade Optical Flow between previous and current frame:

```python3
def sparse_optical_flow(cap: cv2.VideoCapture, out: cv2.VideoWriter, 
                        ShiTomasi_params: dict, pyrLK_params: dict) -> None:
    """
    Sparse optical flow with only 1 Shi-Tomasi Corner Detection and 
    computation via Lucas-Kanade Optical Flow between previous and current frame.
    """
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi Corner Detection
    corners = cv2.goodFeaturesToTrack(old_gray, **ShiTomasi_params)

    # mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # list of random colors
    color = np.random.randint(0, 255, (100, 3))

    # Lucas-Kanade Optical Flow
    ret, frame = cap.read()
    while(ret):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        # nextPts = 2D next points
        # st = status vector, 1 if the the corresponding features has been found
        nextPts, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **pyrLK_params)
        
        # Select good points based on status
        if nextPts is not None:
            good_new = nextPts[st==1]
            good_old = corners[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):

            # reuse the color vector, if it is too short
            if i >= len(color):
                  i %= 10 

            a, b = new.ravel()
            c, d = old.ravel()
            pt1, pt2 = (int(a), int(b)), (int(c), int(d))
            mask = cv2.line(mask, pt1, pt2, color[i].tolist())
        img = cv2.add(frame, mask)
        
        # write the flipped frame
        out.write(img)
        
        # update the previous frame and previous points
        old_gray = frame_gray.copy()
        corners = good_new.reshape(-1, 1, 2)

        # read next frame
        ret, frame = cap.read()
        
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
```

Sparse optical flow with Shi-Tomasi Corner Detection updating in every five frames. Lucas-Kanade Optical Flow computation done between the previous and current frame and the current and previous frame. We use threshold filtering after Euclidian distance computation between two optical flows to choose appropriate tracking points. We have also manipulated a mask and skipped the upper third of a region of interest because the movement in the sky is not expected:

```python3
def sparse_optical_flow2(cam: cv2.VideoCapture, out: cv2.VideoWriter,
        ShiTomasi_params: dict, pyrLK_params: dict,
        frame_interval: int=5, good_threshold:int=1) -> None:
    """ 
    Sparse optical flow with Shi-Tomasi Corner Detection updating in every five frames. 
    Lucas-Kanade Optical Flow computation done between the previous and current frame 
    and the current and previous frame. We use threshold filtering after Euclidian distance 
    computation between two optical flows to choose appropriate tracking points.
    We have also manipulated a mask and skipped the upper third of a region of interest 
    because the movement in the sky is not expected.
    """
    frame_counter = 0
    tracks = list()

    # first frame
    ret, old_frame = cam.read()
    prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # next frame
    ret, frame = cam.read()
    while ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # update the existing tracks by performing optical flow analysis
        if len(tracks) > 0:
            # array of shape (num_tracks, 1, 2) containing the last known positions of each track
            p0 = np.float32([track[-1] for track in tracks]).reshape(-1, 1, 2)
            
            # optical flow between previous and next frame with p0 as the starting point
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **pyrLK_params)
            
            # check the correctness of the calculated optical flow
            # optical flow between next and previous frame with p1 as the starting point
            p0_check, _st, _err = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **pyrLK_params)

            # Euclidean distance d between p0 and p0_check for each track, 
            # store it in an array of shape (num_tracks,)
            # check if the calculated optical flow is accurate
            # if the distance is too large, it indicates that the optical flow 
            # calculation is incorrect and the track should be discarded            
            d = abs(p0 - p0_check).reshape(-1, 2).max(-1)

            # boolean array of shape (num_tracks,)
            # good[i] is True if the distance d[i] is less than a threshold value of 1, 
            # indicating that the optical flow calculation is accurate
            # good is used to filter out tracks that have inaccurate optical flow.
            good = d < good_threshold

            new_tracks = []

            # loop through each track and its new position p1, and good flag
            for track, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                # skip tracks not corresponding with threshold
                if not good_flag:
                    continue
                # if good is True, append the new position to the track
                track.append((x, y))
                # delete the oldest position if the track length exceeds the number of tracks.
                if len(track) > len(tracks):
                    del track[0]
                new_tracks.append(track)
            tracks = new_tracks
            
            # connects the points in each track
            cv2.polylines(frame, [np.int32(tr) for tr in tracks], False, (0, 255, 0), thickness=1)

        # update ShiTomasi corner detection
        if frame_counter % frame_interval == 0:
            mask = np.zeros_like(frame_gray)

            # mask with skipped ROI 1/3 from up(movement in the sky is not expected)
            frame_height = int(cam.get(4))
            mask[int(frame_height / 3):][:] = 255
            corners = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **ShiTomasi_params)
            
            # update points in tracker
            if corners is not None:
                #tracks = list()
                for x, y in np.float32(corners).reshape(-1, 2):
                    tracks.append([(x, y)])

        frame_counter += 1
        prev_gray = frame_gray

        # next frame
        out.write(frame)
        ret, frame = cam.read()

    # Release everything if job is finished
    cam.release()
    out.release()
    cv2.destroyAllWindows()
```

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
