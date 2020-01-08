import numpy as np
import matplotlib.pyplot as plt
import tanuki_ml

sample = 255*np.array([ [0,0,0,1,0,0,0,0],
                        [1,1,0,0,1,0,0,0],
                        [0,0,1,0,0,1,0,0],
                        [0,0,0,1,0,0,1,0],
                        [0,0,0,1,0,0,1,0],
                        [0,0,0,1,0,0,1,0],
                        [0,0,0,1,0,0,1,0]])
sample = sample[..., np.newaxis]

m1 = tanuki_ml.path_determiner()
paths = m1.approx_path(sample)
terminals = m1.get_terminal_point(paths)
idxs=m1.draw_line(terminals[0], terminals[1])

img = np.zeros_like(sample)
for i in idxs:
    img[i[0],i[1]]= 255

plt.imshow(img[...,0],cmap="gray")
plt.show()