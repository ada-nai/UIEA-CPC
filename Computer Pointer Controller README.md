# Computer  Pointer Controller

## 1. Introduction/Description
- The Computer Pointer Controller App uses multiple pre-trained models to run inferences using the OpenVINO toolkit to control the mouse pointer of a computer
- There are four models which are used in tandem to make inferences and control the mouse pointer depending on the gaze estimation of the user's eyes
- The four pre-trained models used in this project are:  
i. [Face Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)  
ii. [Landmark Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)  
iii. [Head Pose Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)  
iv. [Gaze Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)  

![CPC pipeline](https://github.com/ada-nai/UIEA-CPC/blob/master/cpc_pipeline.jpg?raw=true)

## 2. Project Setup and Installation
1. Download the [Intel® Distribution of OpenVINO™ toolkit for Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html). Python `v3.6` is recommended.
2. Unzip contents of the project archive in a directory (e.g.: under /home/ user_name /Public). Now a folder `CPC` would be present after extraction
3. Open a terminal window after navigating to the project folder `CPC` and create a virtual environment   

    a. __~/Public/CPC $__ `python3 -m venv cpc`  

    b. The virtual environment is now created
    
    c. Now a `cpc` sub-folder would have been created within the main `CPC` project folder, which contains all python interpreter files
    
4. Activate the virtual environment  
    
    a. `source <env_name>/bin/activate` (Here, `<env_name>` is `cpc`)
    
5. Install the required packages using the `requirements.txt` file  
    
    a. Enter `pip3 install -r requirements.txt`
6. Download the required pre-trained models using `downloader.py`  
    
    a. To navigate to the directory containing the Model Downloader: `cd
/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader`  

    b. Download the Face Detection model:   
    `sudo ./downloader.py --name face-detection-adas-binary-0001 -o <project directory>`  
    
    c. Download the Landmark Detection model:   
    `sudo ./downloader.py --name landmarks-regression-retail-0009 -o <project directory>`  
    
    d. Download the Head Pose Estimation model:   
    `sudo ./downloader.py --name head-pose-estimation-adas-0001 -o <project directory>`  
    
    e. Download the Gaze Estimation model:   
    `sudo ./downloader.py --name gaze-estimation-adas-0002 -o <project directory>`  
    

#### NB: 
1. The `<project directory>` should be replaced by the path of the root directory as mentioned in the File Structure section below in Section 5
2. Each model will be downloaded under an `intel` folder in the root directory as shown in the File Structure as shown below. In the end the file structure of the project directory at root must look like how it is shown below in Section 5

## 3. How to run a demo
1. Open a terminal window
2. Navigate to the project folder (**Make sure to run commands from the root folder (unless specified), here `/home/ user_name /Public/CPC`**)
3. Activate the virtual environment  

    a. Enter `source <env_name>/bin/activate` in the terminal (Here, <env_name> is `cpc`)  
    
4. Activate the environment variables for OpenVINO  

    a. Enter `cat vinovar.txt` in the terminal. It should display the command to activate the OpenVINO environment variables:  
    `source /opt/intel/openvino/bin/setupvars.sh`  
    
    b. Copy and paste the command in the terminal  
    
    c. The system has now been setup to run the project  
    
5. Run the program in the terminal  

    a. Navigate to the `src` folder using the command `cd src`  
    
    b. Enter the following command in the terminal for a demo:  
    
    `python3 main.py -i demo.mp4 -if video -fd <path to face model> -ld <path to landmark model> -hpe <path to head_pose model> -ge <path to gaze model>`  
    
    c. Command given in step 5a is a basic run of the app. The app can be run with a number of options as mentioned in Step 4. You may choose to include them.  
    
    d. The user can exit the app at any given moment by pressing the `X` key of the keyboard  
    
#### NB: The paths for the models are not necessarily required to be provided. If the file structure is followed as in Section 5, and the necessary files are present for FP32 precision of all models, the app will run using the FP32 precision variant of all the models to run the app

## 4. Command Line Interface options
The Computer Pointer Controller App allows the user to enable a number of options. The user can check the options by entering `python3 main.py -h` in the terminal window

The options available for user to configure the CPC app are as follows:
1.   `-h`, --help            show this help message and exit
2.   `-i` INPUT, --input INPUT
                        Path of video file, if applicable
2.   `-l` EXTENSION, --extension EXTENSION
                        MKLDNN targeted custom layers.Absolute path to a
                        shared library with the kernels implementation.
4.   `-d` DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
5.   `-if` INPUT_TYPE, --input_type INPUT_TYPE
                        Input media format to the models. 'cam' or 'video'.
                        Default set to 'video'
6.   `-fd` FACE, --face FACE
                        Path of Face Detection model xml file
7.   `-ld` LANDMARK, --landmark LANDMARK
                        Path of Landmark Detection model xml file
8.   `-hpe` HEAD_POSE, --head_pose HEAD_POSE
                        Path of Head Pose Estimation model xml file
9.   `-ge` GAZE, --gaze GAZE
                        Path of Gaze Estimation model xml file
10.  `-vf` VISUAL_FLAG, --visual_flag VISUAL_FLAG
                        Flag for visualization of model outputs. Can be 0 or
                        1. Default set to 0.
11.  `-pf` PERF_FLAG, --perf_flag PERF_FLAG
                        Flag for analyzing layer-wise performance of models.
                        Can be 0 or 1. Default set to 0.



## 5. Explanation of the directory structure and overview of the files used in the project

#### NB: The sub-folders of the `cpc` and `__pycache__` folder have not been included as it contains files of the virtual environment and compiled files respectively, and the sub-directories for the same can be ignored

```
(cpc) doctor_s@doc-Predator:~/Public/cpc-uiea$ tree -I 'cpc|starter'  

<root>
├── cpc
├── intel
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP32
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32-INT8
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP32
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32-INT8
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP32
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32-INT8
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── README.md
├── requirements.txt
├── src
│   ├── CPC.log
│   ├── demo.mp4
│   ├── face.py
│   ├── gaze.py
│   ├── head_pose.py
│   ├── input_feeder.py
│   ├── landmark.py
│   ├── main.py
│   ├── model_perf
│   ├── mouse_controller.py
│   ├── __pycache__
└── vinovar.txt
```

## 6. Benchmarks
The different time related parameters were measured and the comparison of times between FP32 and FP16 models is shown below:

![Timing Benchmark](https://raw.githubusercontent.com/ada-nai/UIEA-CPC/fin/Timing%20Models.png)

- We can observe that the model loading time and the average processing time for models of FP16 precision are significantly lower than the FP32 precision counterparts
- This is because the model weights are of lower precision and occupy lesser memory and are thus faster to load
- However, there was not much difference observed in the inference times, moreover because the observations were made on the same hardware ('CPU')
- The FP16 model was less accurate in terms of the gaze output, though it gave better performance in time benchmarks. This is because some precision is lost and the weights are not as significant as their FP32 counterpart models

## 7. Stand Out Suggestions

### A. Logging and Error Handling
- The project code demonstrates logging and exception handling to a notable level. 
- All key events and errors are dumped into the `CPC.log` file in the `src` folder. 
- The user can check for key events and errors by referring the same file. 
- Additionally, key time-related benchmark stats are also logged at the end of the log file. The user can check them out too under the title `---APP STATS---`

### B. Choice of input file format
- The app gives the user the choice to select the input format of their choice
- They can choose from either a video file or a webcam feed
- This option can be chosen using the `-if` argument when running the program

### C. Visualizations
- The app allows the user to visualize the outputs from the intermediate models in the pipeline to inquisitively understand the flow/working of the logic
- They can do so by setting the `-vf` argument to `1`
- It must be noted that the default value is 0 and the outputs cannot be visualized unless the user explicitly sets the flag value
- The outputs that can be visualized are: 
    a. The detection of the face from the Face Detection Model
    b. The cropped eyes (left, right)
    c. The head_pose output (x, y axes)

![Visualized Outputs](https://raw.githubusercontent.com/ada-nai/UIEA-CPC/master/visualized_outputs.png)

### D. Parsing of Command Line Arguments
- The code uses command-line arguments to change the behavior of the program. For instance, specifying model file, specifying hardware type, etc.
-  Default arguments are used for when the user does not specify the arguments, in some cases. This even includes the default paths for all the models, and the user need not bother adding paths, assuming the files are present in the path as mentioned in Section 5 and they want to perform inference using FP32 models
-  A `-h or --help` option is present and displays information about the different configurations for the code

### E. Benchmarking Layer-wise performance
- The code allows user to view the time taken for execution of all `EXECUTED` layers for all 4 models
- This can be done by setting the `-pf` argument flag to 1. The code uses the `get_perf_counts` function provided under the `IECore` API to display layer-wise performance for all models
- Once the app has completed execution, the user can view these stats in the respective files in the `model_perf` folder
- Four files would be available containing benchmark results for the four models:
    i. face.txt
    ii. landmark.txt
    iii. head_pose.txt
    iv. gaze.txt
- The files are overwritten with each instance of running the app

### F. Edge Cases
- One of the edge cases is where the input to the gaze estimation model has invalid dimensions (one of the width/height parameters is 0), causing cv2 to throw an error because the frame cannot be reshaped. For this reason, such frames are ignored.



















