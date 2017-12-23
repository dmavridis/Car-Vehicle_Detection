# Import necessary functions and packages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import itertools
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from vehicle_detection_functions import *
from heatmap_functions import *

#%% Load pickle file
# If it does not exist run the script vehicle_train_classifier

dist_pickle = pickle.load(open("classifier/svc_pickle_YUV.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"] 
hog_feat = dist_pickle["hog_feat"]



#%% Define searching windows
windows_list = []
ystart = 400
ystop = 656

# Use a random image to get the windows
image = mpimg.imread('test_images/test1.jpg')

#Level 1: Small window
windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[ystart, ystart + 96], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows_list.append(windows)

#Level 2: Medium window
windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[ystart, ystop], 
                       xy_window=(128, 128), xy_overlap=(0.5, 0.5))
windows_list.append(windows)

#Level 3: Large window
windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[ystart, ystop],
                       xy_window=(160, 160), xy_overlap=(0.5, 0.5))
windows_list.append(windows)

windows = list(itertools.chain(*windows_list))

#%%
plt.imshow(draw_boxes(image,windows))

#%% Pipeline
def tracking_pipeline(image):
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

#%%

image = mpimg.imread('test_images/test5.jpg')
draw_img = tracking_pipeline(image)
#out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#fig = plt.figure()
#plt.imshow(draw_img)
#plt.title('Car Positions')

test_images = []
for ii in range(1,7):
    image = mpimg.imread('test_images/test' + str(ii) + '.jpg')
    draw_img = tracking_pipeline(image)
    test_images.append(draw_img)
#    plt.imshow(draw_img)

fig = plt.figure()
plt.subplot(321)
plt.imshow(test_images[0])
plt.subplot(322)
plt.imshow(test_images[1])
plt.subplot(323)
plt.imshow(test_images[2])
plt.subplot(324)
plt.imshow(test_images[3])
plt.subplot(325)
plt.imshow(test_images[4])
plt.subplot(326)
plt.imshow(test_images[5])
