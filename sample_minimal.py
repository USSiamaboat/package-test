"""
Minimal example of using the mrc_parser example
"""

# This is all you need to import
from mrc_parser import MRC_Parser, BoxnetTF

# Config
mrc_path = "new-data/thing.mrc"
model_path = "boxnet.tflite"

# Parser init in one line
parser = MRC_Parser(BoxnetTF(model_path))

# Parse in one line
result = parser(mrc_path)

# Access softmax channels like this
# All of the following variables are 2D np arrays with the same dimensions as the input image
none_channel = result[:, :, 0]
particle_channel = result[:, :, 1]
dirty_things_channel = result[:, :, 2]
