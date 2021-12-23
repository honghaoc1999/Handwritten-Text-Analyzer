import numpy as np
import scipy.io
from nn import *
import copy
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
# print(valid_y)

max_iters = 50
# pick a batch size, learning rate
batch_size = 600
learning_rate = 0.0015
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
print(batch_num)

params = {}

print(train_y.shape, train_x.shape)
print(valid_x[1])
# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')
initial_layer1_w = copy.deepcopy(params['Wlayer1'])

train_accuracies = []
train_losses = []
valid_accuracies = []
valid_losses = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # pass
        # training loop can be exactly the same as q2!
        # forward
        post_act = forward(xb,params,'layer1')
        probs = forward(post_act,params,'output',softmax)
        # loss
        (loss, accur) = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        total_acc += accur
        # backward
        delta1 = probs
        yb_idx = []
        for i in range(yb.shape[0]):
            yb_idx.append(np.argmax(yb[i]))
            
            
        delta1[np.arange(probs.shape[0]),yb_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        # Implement backwards!
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        grad_W_output, grad_b_output = params['grad_Woutput'], params['grad_boutput']
        params['Woutput'] -= learning_rate*grad_W_output
        params['boutput'] -= learning_rate*grad_b_output
        grad_W_layer1, grad_b_layer1 = params['grad_Wlayer1'], params['grad_blayer1']
        params['Wlayer1'] -= learning_rate*grad_W_layer1
        params['blayer1'] -= learning_rate*grad_b_layer1
    total_acc = total_acc/float(batch_num)
    train_accuracies.append(total_acc)
    train_losses.append(total_loss)
    # params_for_train = copy.deepcopy(params)
    # params_for_valid = copy.deepcopy(params)
    # post_act = forward(train_x,params,'layer1')
    # probs = forward(post_act,params,'output',softmax)
    # # loss
    # (loss, accur) = compute_loss_and_acc(train_y, probs)
    # train_accuracies.append(accur)
    # train_losses.append(loss)
    post_act = forward(valid_x,params,'layer1')
    probs = forward(post_act,params,'output',softmax)
    # loss
    (loss, accur) = compute_loss_and_acc(valid_y, probs)
    valid_accuracies.append(accur)
    valid_losses.append(loss)
    
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.plot(range(50), valid_accuracies, label = "valid accur") 
plt.plot(range(50), train_accuracies, label = "train accur") 
plt.legend() 
plt.show()
plt.plot(range(50), valid_losses, label = "valid loss") 
plt.plot(range(50), train_losses, label = "train loss") 
plt.legend() 
plt.show()
# run on validation set and report accuracy! should be above 75%
valid_acc = None
post_act = forward(valid_x,params,'layer1')
probs = forward(post_act,params,'output',softmax)
print(probs.shape, valid_y.shape)
# loss
(loss, valid_acc) = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
post_act = forward(test_x,params,'layer1')
probs = forward(post_act,params,'output',softmax)
# loss
(loss, test_acc) = compute_loss_and_acc(test_y, probs)
print("test_acc", test_acc)

# Q3.3
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
trained_layer1_w = params['Wlayer1']

fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )
for i in range(64):
    print(initial_layer1_w[:, i].shape)
    grid[i].imshow(np.reshape(initial_layer1_w[:, i], (32, 32)))
plt.show()
fig = plt.figure(1, (4., 4.))
grid1 = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )
for i in range(64):
    print(trained_layer1_w[:, i].shape)
    grid1[i].imshow(np.reshape(trained_layer1_w[:, i], (32, 32)))
plt.show()
# print(initial_layer1_w.shape, train_x.shape, trained_layer1_w.shape)

# plt.imshow(initial_layer1_w)
# plt.grid(True)
# plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
post_act = forward(test_x,params,'layer1')
probs = forward(post_act,params,'output',softmax)

for i in range(probs.shape[0]):
    actual_label = np.argmax(test_y[i])
    pred_label = np.argmax(probs[i])
    # if actual_label != pred_label:
    confusion_matrix[actual_label, pred_label] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
