# Road Sign and Traffic Light Detection
<div align = 'center'> 
	<img src ='https://github.com/mchockal/road_sign_traffic_light_detection/blob/main/result_use_cases.png' alt="Road sign and traffic light detection under different use cases">
</div>

## 1. Description:
Detect and classify road signs and traffic lights without using deep learning \
This will use sample input images and generate output images to directory passed as arg. \
Follow the instructions added as comments in `main.py` to learn how to run the code for the larger/ any other dataset. \
\
Note that the larger subset of LISA dataset used to generate the results seen is not available to avoid plaigarism.
The code for `lights_classifier` and `sign_classifier` training method, dataset used are also redacted for same reasons.
The approach used is described towards the end of this file to develop an intuition if you are solving the same problem.

## 2. Sample code run:
`python main.py --t light --i ./Resources/Dataset/lightss --o ./output` 

## 3. Environment setup:
All package requirements are frozen using pip freeze to `requirements.txt`. 
Make sure all relevant packages are available in your python venv or conda env before executing the code.

## 4. Directory structure:
- `output`
  - `lights` : Output images (and videos) of traffic light detector. Has annotated images used in the report.
  - `signs` : Output images of traffic sign recognizer. Has annotated images used in the report.

- `Resources`
  - `Dataset`
    - `Lights` : Empty directory. This is where the LISA light subset from box should be downloaded to.
		- `lightss` : Sample traffic light images for input
		- `Signs` : Sample traffic sign images for input
    - `Lights`
      - `go` : Training images for SVM and bow lights classifier
      -  `stop` : Training images for SVM and bow lights classifier
    - `Signs`
      - `pedestrianCrossing` : Training images for SVM and bow sign classifier
      -  `signalAhead` : Training images for SVM and bow sign classifier
      -  `stop` : Training images for SVM and bow sign classifier
			
## 5. Approach:
The implemented model is based off the ideas drawn from multiple researches in the domain. \
Two models are developed - one for traffic light detection and another for traffic
sign recognition. \ 
For each of the traffic light and sign recognition models, there are two phases.
- Object detection
- Object classification
  
Both traffic sign and light detection have similar approach for detection and classification, hence the common approach is described below categorized under the two phases.

### 5.1 Object Detection
For each frame in the video, the input frame is first resized to 640 x 480 and converted to HSV
format. Color thresholds are applied accordingly for traffic lights (red, green) and signs (red, yellow).
The thresholded image is then subject to morphological operations - erosion followed by dilation.
The first erosion removes noisy pixels in the background and only the candidate regions with high
probability remains. The dilation that follows adds more surface area to the eroded pixels to make
sure that the necessary regions of interests are not filtered in the following steps due to *heuristic-based
thresholding*. Contours are then detected in the thresholded image and blobs that don’t fit the pre
defined heuristics for traffic light (or sign) are ignored. Thus, only limited regions of interests are
retained within bounding boxes, which are then passed to the classifier for light (or sign) recognition.

### 5.2 Object Classification
The initial attempt at classification involved extracting SIFT features from the regions of interests
obtained in the detection phase and compare them against a dictionary of SIFT features for each of
the traffic sign. The one with the best match is returned as the result. However, this did not work well
for traffic lights and was not very robust for signs too.

The improved approach involved extracting the SIFT features, group them into bag of visual words
(or features) and use them for prediction using a SVM classifier trained on the bag of visual features
for each sign (or traffic light for light recognition). Since this approach involves learning to predict features, the
system should be trained on the different classes prior to prediction. Two different classifiers are
trained...
- Traffic sign classifier
- Traffic light classifier. 

The training and prediction steps involved are the same for both classifier and the steps are detailed in the next two sections.

#### 5.2.1 Training
The training phase for both classifier includes vocabulary generation (K-Means clustering of SIFT
features) using bag of visual words, and then training a SVM classifier with the clustered feature
vectors and their corresponding object (sign type or traffic light) as classes or labels. Two classifier
models (*sign_classifier.pkl*, *lights_classifier.pkl*) and two vocabularies (*sign_classifier_vocab.npy*,
*lights_classifier_vocab.npy*) are saved at the end of training phase. These are loaded during prediction
time in the first frame processing step.

#### 5.2.2 Inference
During inference, the SIFT features are extracted from the input image and resized to the dimensions
the classifier is trained on. A bag of visual words object is initialized with the pre-trained vocabulary
for signs (or lights) obtained from training phase. The extracted SIFT features are grouped into
clusters using the respective bag of visual words object. These clusters are passed as input to the
corresponding SVM classifier which has been pre-trained on different signs ( or lights). The classifier
returns the prediction back to the main pipeline. 

An interesting advantage of this approach is that since SIFT features are used for both training and
prediction, for regions where there is less shift in gradients (potential false candidates), the feature
vector will be returned as an empty array. This eliminates false candidates even before the input is
passed to the actual classifier.

