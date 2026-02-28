import sys, os
sys.path.append(os.path.abspath('.'))
from src.custom_gaze_tracker import CustomPyTorchGazeTracker
print('Testing initialization...')
tracker = CustomPyTorchGazeTracker()
print('Success!')
