import numpy as np
import numpy as np

random_array_5_20 = np.random.rand(2, 2)

padded_array_20_20 = np.zeros((3, 3))

padded_array_20_20[:2, :2] = random_array_5_20

print(random_array_5_20)
print(padded_array_20_20)
