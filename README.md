To Set Up the conda Virtual Environment:

conda create -n yolov5-env python=3.8

conda activate yolov5-env

conda install pytorch torchvision torchaudio -c pytorch

conda install -c conda-forge opencv

pip install git+https://github.com/ultralytics/yolov5.git
