import sys, os 

sys.path.append('/usr/local/lib/python3.6/site-packages/')
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device_name = pynvml.nvmlDeviceGetName(handle)

if device_name != b'Tesla T4':
  raise Exception("""
    Unfortunately Colab didn't give you a T4 GPU.
    
    Make sure you've configured Colab to request a GPU instance type.
    
    If you get a K80 GPU, try Runtime -> Reset all runtimes...
  """)
else:
  print('***********************************')
  print('Woo! You got the right kind of GPU!')
  print('***********************************')
  print()
