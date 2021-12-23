import numpy as np

# use for a "no activation" layer
def linear(x):
    return x

def linear_deriv(post_act):
    return np.ones_like(post_act)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(post_act):
    return 1-post_act**2

def relu(x):
    return np.maximum(x,0)

def relu_deriv(x):
    return (x > 0).astype(np.float)

def func(i):
    np.argmax(i)

# mat = np.array([[0,1,0],[1,0,1],[0,0,1],[2,3,1]])
# arr = np.array([0,1,0])
# print(mat+arr)

# vfunc = np.vectorize(func)
# print(np.argmax(mat[1]))
# arr = np.arange(40)
# print(np.random.shuffle(arr))

# mat = np.array([1,2,3,4,5,6,7])
# print(np.reshape(mat[1:], (2,3)))
# in_size = 20
# out_size = 10
# w = np.random.uniform(-np.sqrt(6.0)/np.sqrt(in_size+out_size),np.sqrt(6.0)/np.sqrt(in_size+out_size),out_size)
# print(w)