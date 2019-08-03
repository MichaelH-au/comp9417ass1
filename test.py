import numpy as np

a = np.array([[5,35],[25, 279]])
b = np.array([[279/170, -7 / 34],[-7 / 34, 1 / 34]])

inv = np.linalg.inv(a)

x = np.array([[1,1,1,1,1], [3,6,7,8,11]])
aInv = x.transpose()
print(aInv)
print(np.dot(x, aInv))
y = np.array([13, 8, 11, 2, 6])

xy = np.dot(x, y)

print(np.dot(b, xy))

