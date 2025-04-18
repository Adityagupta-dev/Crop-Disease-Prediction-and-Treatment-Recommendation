import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
print(f"TF: {tf.__version__}")
print(f"CUDA: {tf.sysconfig.get_build_info()['cuda_version']}")
print(f"cuDNN: {tf.sysconfig.get_build_info()['cudnn_version']}")
print("GPUs:", tf.config.list_physical_devices('GPU'))