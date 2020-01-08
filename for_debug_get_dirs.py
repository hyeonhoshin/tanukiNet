import numpy as np
import matplotlib.pyplot as plt
import tanuki_ml
import pickle
import cv2
from skimage import feature, transform

'''
sample = 255*np.array([ [0,0,0,1,1,0,0,0,0,0,0],
                        [1,1,0,0,1,1,0,0,0,1,0],
                        [0,1,1,0,0,1,1,0,0,1,0],
                        [0,0,1,1,0,0,1,1,0,1,0],
                        [0,0,1,1,0,0,1,1,0,1,0],
                        [0,0,1,1,0,0,1,1,1,1,0],
                        [0,0,1,1,0,0,1,1,0,0,1]])
                        '''
with open("a_lane.p","rb") as f:
    sample = pickle.load(f)[...,0]

sample = feature.canny(sample,sigma=1,high_threshold=150,low_threshold=50)
plt.imshow(sample,cmap='gray')
plt.show()

lines = transform.probabilistic_hough_line(sample,line_length=50)
theta = []
for line in lines:
    p1, p2 = line
    theta.append(np.arctan((p2[1]-p1[1])/(p2[0]-p1[0])))

priority = np.argsort(theta)
idx1,idx2 = priority[0],priority[-1]

selected_lines = [lines[idx1],lines[idx2]]

plt.imshow(sample*0,cmap='gray')

for line in selected_lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

plt.show()

l1 = selected_lines[0]
l2 = selected_lines[1]

s1,e1 = l1
s2,e2 = l2

# start pt
s_mid = ((s1[0]+s2[0])//2, (s1[1]+s2[1])//2)
e_mid = ((e1[0]+e2[0])//2, (e1[1]+e2[1])//2)
l_mid = s_mid, e_mid

plt.imshow(sample*0,cmap='gray')
p0, p1 = l_mid
plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.show()