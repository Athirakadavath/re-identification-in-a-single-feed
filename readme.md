# Reidentification in a Single Feed

This repository contains code for player (and ball/referee) detection and tracking using YOLO models in sports video feeds.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV
- Other dependencies in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Model Weights

**Note:**
The YOLO model weights (`models/best.pt`) are **NOT included in this repository** because they exceed GitHub's 100MB file size limit.

#### How to obtain the weights

1. Download the `best.pt` model weights from the following link (choose one):

   - Google Drive: [Model link](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
   
2. Place the downloaded `best.pt` file in the `models/` directory:

   ```
   models/best.pt
   ```

### Usage

Run the detection and tracking script:

```bash
python src/detect.py --video data/15sec_input_720p.mp4 --weights models/best.pt
```

Output video will be saved as `output.mp4` in your project folder.

### Model Weights Directory

If the `models/` folder does not exist, create it:

```bash
mkdir models
```

### Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)


**If you have questions or issues, please open an Issue or Discussion!**
