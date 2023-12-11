## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### Environment using: Python 3.8, torch==1.9.0, cudatoolkit=11.1  
```bash
pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python

# it is recommended to install wheel
pip install --upgrade wheel (--user)

# Install detectron2 as package, under your project location (i.e. working directory)
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .

# Generally the structure of the project will be something as:

/working_directory

# Install and download COCO dataset
pip install git+https://github.com/cocodataset/panopticapi.git
# At this stage you need to download the coco dataset from their official website

# It is recommened to download all regarding COCO 2017 dataset, which means:
# train2017, eval2017, test2017 recalling that you need to insert those in
# '/detectron2/datasets/coco/annotations'

# Install and download Citiscrapes
pip install git+https://github.com/mcordts/cityscapesScripts.git

# Now, we can replicate our project cloning it into our working directory
# Let's go back to the previous and current
cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

```code

#### Run properly your training
`You need to manually edit '__init__' version from your original into 
import tensorboard
from distutils.version import LooseVersion

if not hasattr(tensorboard, '__version__') or LooseVersion(tensorboard.__version__) < LooseVersion('1.15'):
raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')

del LooseVersion
del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
```
