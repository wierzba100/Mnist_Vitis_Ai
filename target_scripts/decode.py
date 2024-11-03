import numpy as np

file_path = '0.Net__Net_Linear_fc2__ret_fix.bin'

data_type = np.int8

data = np.fromfile(file_path, dtype=data_type)

print(data)
