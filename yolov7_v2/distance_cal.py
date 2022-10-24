import random as rd
import numpy as np
import math
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def function(x,y):
    b = (
        (x[1]**2-x[2]**2)*(y[0]-y[1])-(x[0]**2-x[1]**2)*(y[1]-y[2])
        )/(
        (x[1]**2-x[2]**2)*(x[0]-x[1])-(x[0]**2-x[1]**2)*(x[1]-x[2])
        )
    a = (y[0] - y[1] - b*(x[0]-x[1]))/(x[0]**2-x[1]**2)
    c = y[0] - a*x[0]**2 - b* x[0]
    return a,b,c

with open('./heights.txt','r') as f:
    x = np.array([int(x) for x in f.readlines()]).reshape(-1,4)
with open('./dis.txt','r') as f:
    y = np.array([float(y) for x in f.readlines() for y in x.split(',')]).reshape(-1,2)

dis = np.array([math.sqrt(z[0]**2+z[1]**2) for z in y]).reshape(18,1).tolist()
ran_idx =  sorted(set([dis.index(temp) for temp in rd.choices(dis,k = 3)]))

# print('ran_idx is :',ran_idx)

# # for _ in range(5):
# target_distance = np.array([x for x in [(dis[ran_idx[0]]),(dis[ran_idx[1]]),(dis[ran_idx[2]])]])
# target_height = np.array([x for x in [np.mean(x[ran_idx[0]]),np.mean(x[ran_idx[1]]),np.mean(x[ran_idx[2]])]])
# print(target_distance)
# print(target_height)
# ### x == distance y == LiDAR points(x,y)
# a,b,c =function(target_distance,target_height)
# print(a,b,c)

print(dis)
print(len(dis))
print(x)

# print(rd.choices(x,k=3)[0])
# print(rd.choices(rd.choices(x,k=3)[0],k=1))
# loss =  a * test_height**2 +b* test_height  + c


save_txt = './distance.txt'
# print([x+'\n' for x in re.split('[ ] ,',str(self.height))])
for i,distance in enumerate(dis):
    if i == 0:
        with open(save_txt,'w') as f:
            f.write(str(distance[0]) + '\n')
    else:
        with open(save_txt,'a') as f:
            f.write(str(distance[0]) + '\n')


# plt.plot(target_height, target_distance, label='real curve')
# # plt.plot(target_height, a * target_height**2 + b* target_height  + c, 'bo', label='noise curve')
# plt.legend()
# plt.show()

