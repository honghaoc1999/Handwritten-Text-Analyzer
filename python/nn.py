import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = np.zeros((in_size, out_size)), np.zeros(out_size)
    # print(in_size, out_size)
    for i in range(W.shape[0]):
        W[i] = np.random.uniform(-np.sqrt(6.0)/np.sqrt(in_size+out_size),np.sqrt(6.0)/np.sqrt(in_size+out_size),out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
        # c = np.max(x[i])
        # offset_exp_row = np.exp(x[i] - c)
        # sum = np.sum(offset_exp_row)
        # x[i] = offset_exp_row/np.sum(offset_exp_row)
    res = 1.0/(1.0+np.exp(-x))
    return res
    # for i in range(x.shape[0]):
    #     c = np.max(x[i])
    #     offset_exp_row = np.exp(x[i] - c)
    #     sum = np.sum(offset_exp_row)
    #     x[i] = offset_exp_row/np.sum(offset_exp_row)
    # res = x
    # return res

# Q 2.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    # print(W.shape, b.shape)
    pre_act = np.zeros((X.shape[0], b.shape[0]))
    for i in range(X.shape[0]):
        pre_act[i] = np.dot(W.T, X[i]) + b
    # pre_act = np.dot(W.T, X) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    for i in range(x.shape[0]):
        c = np.max(x[i])
        offset_exp_row = np.exp(x[i] - c)
        sum = np.sum(offset_exp_row)
        x[i] = offset_exp_row/np.sum(offset_exp_row)
    res = x
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    loss = -np.sum(y * np.log(probs))
    err_cnt = 0
    for i in range(probs.shape[0]):
        actual_label = np.argmax(y[i])
        pred_label = np.argmax(probs[i])
        if actual_label != pred_label:
            err_cnt += 1
    acc = (float(probs.shape[0])-float(err_cnt))/float(probs.shape[0])
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 2.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    act_delta = delta * activation_deriv(post_act)
    grad_b = np.zeros(b.shape)
    # print("delta:",act_delta.shape)
    # print(act_delta[0].shape, grad_b.shape)
    for i in range(act_delta.shape[0]):
        grad_b += act_delta[i]
    
    # print("yea", b.shape, grad_b.shape, act_delta)
    grad_W = np.dot(X.T, (act_delta))
    grad_X = np.dot(act_delta, W.T)
    # print(X.shape[0])
    
    

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    random_indices = np.arange(x.shape[0])
    np.random.shuffle(random_indices)
    (curr_x_batch, curr_y_batch) = np.zeros((batch_size, x.shape[1])), np.zeros((batch_size, y.shape[1]))
    offset = x.shape[0]%batch_size
    # print(random_indices)
    for i in range(x.shape[0]-offset):
        curr_x_batch[i%batch_size] = x[random_indices[i]]
        curr_y_batch[i%batch_size] = y[random_indices[i]]
        # print(x[random_indices[i]])
        if i%batch_size == batch_size-1:
            # print((curr_x_batch, curr_y_batch))
            batches.append((curr_x_batch, curr_y_batch))
    if offset != 0:
        (curr_x_batch, curr_y_batch) = np.zeros((offset, x.shape[1])), np.zeros((offset, y.shape[1]))
        for i in range(offset):
            curr_x_batch[i] = x[random_indices[x.shape[0]-offset+i]]
            curr_y_batch[i] = y[random_indices[y.shape[0]-offset+i]]
        batches.append((curr_x_batch, curr_y_batch))
 
    return batches
