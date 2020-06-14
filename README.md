## Structure of Directory
 tree -I 'cpc'
.
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
│   ├── mouse_controller.py
│   ├── __pycache__
│   │   ├── face.cpython-36.pyc
│   │   ├── gaze.cpython-36.pyc
│   │   ├── head_pose.cpython-36.pyc
│   │   ├── input_feeder.cpython-36.pyc
│   │   ├── landmark.cpython-36.pyc
│   │   └── mouse_controller.cpython-36.pyc
│   └── video_test.py
├── starter
│   ├── bin
│   │   └── demo.mp4
│   ├── README.md
│   ├── requirements.txt
│   └── src
│       ├── input_feeder.py
│       ├── model.py
│       └── mouse_controller.py
└── vinovar.txt

20 directories, 45 files
