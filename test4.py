import numpy as np
a = np.load("./logs/opaque/opaque_2/opaque_0_2.npy")
a = a.reshape(800,800)
print(a.shape)
print(a[a==True].shape)