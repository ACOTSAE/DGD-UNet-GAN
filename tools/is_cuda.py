#!/bin/env python
import torch
if not (avail := torch.cuda.is_available()):
    print(f'CUDA is available: {avail}')
    exit()
print(f'CUDA is available: {avail}')
print('CUDA Version:',torch.version.cuda)
print('Pytorch Version:',torch.__version__)
print('Is GPU available:',torch.cuda.is_available())
print('Device Count:',torch.cuda.device_count())
print('Is bf16 Supported:',torch.cuda.is_bf16_supported())
print('Device Name:',torch.cuda.get_device_name())
print('CUDA Compute compatibility:',torch.cuda.get_device_capability())
print('Total GPU Memory:',torch.cuda.get_device_properties(0).total_memory/1024/1024/1024,'GB')
print('Is TensorCore available:',torch.cuda.get_device_properties(0).major >= 7)
