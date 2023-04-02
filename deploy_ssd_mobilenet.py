%%capture
%pip install sagemaker -U

import os
import glob
import cv2
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.utils import name_from_base
%matplotlib inline
import matplotlib.pyplot as plt
import visualization_utils as viz_utils

role = sagemaker.get_execution_role()

# TODO: Update the model artifact here. 
model_artifact = 's3://object-detection-ravi/ssd-mobilenet/model.tar.gz'

model = TensorFlowModel(
    name=name_from_base('tf2-object-detection'),
    model_data=model_artifact,
    role=role,
    framework_version='2.8'
)

predictor = model.deploy(initial_instance_count=1, instance_type='ml.g4dn.xlarge')

frames_path = sorted(glob.glob('../data/test_video/*.png'), 
                     key = lambda k: int(os.path.basename(k).split('.')[0].split('_')[1]))

import numpy as np
def load_image(path: str) -> np.ndarray:
    """ This function reads an image from the path and returns a numpy array"""
    cv_img = cv2.imread(path,1).astype('uint8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img

category_index = {
                    1:{'id': 1, 'name': 'vehicle'}, 
                    2: {'id': 2, 'name': 'pedestrian'},
                    4: {'id': 4, 'name': 'cyclist'}
                }

def image_file_to_tensor(path):
    cv_img = cv2.imread(path,1).astype('uint8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img

images = []
for idx, path in enumerate(frames_path):
    if idx % 10 == 0:
        print(f'Processed {idx}/{len(frames_path)} images.')
        
    # load image
    img = image_file_to_tensor(path)
    inputs = {'instances': [img.tolist()]}
    
    # run inference and extract results
    detections = predictor.predict(inputs)['predictions'][0]
    detection_boxes = np.array(detections['detection_boxes'])
    detection_classes = [int(x) for x in detections['detection_classes']]
    detection_scores = detections['detection_scores']
    
    # display results on image
    image_np_with_detections = \
        viz_utils.visualize_boxes_and_labels_on_image_array(
            img,
            detection_boxes,
            detection_classes,
            detection_scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=0.6,
            agnostic_mode=False)
    images.append(image_np_with_detections)

plt.imshow(images[0])

frame_width = images[0].shape[0]
frame_height = images[0].shape[1]

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Read and display the images
for image in images:
    out.write(image) # Write the image to the video
    if cv2.waitKey(1) == ord('q'): # Hit `q` to exit
        break
        
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
