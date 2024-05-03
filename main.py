import torch

x = torch.empty(3, 3, 3, 3)
x = torch.rand(3, 3, 3, 3)
x = torch.zeros(3, 3)
x = torch.ones(3, 3)
x = torch.eye(3, 3)
x = torch.eye(3, 3, dtype=torch.int)
x = torch.tensor([2.5, 0.1])

print(x)
print(x.dtype)
print(x.size())
print(x.shape)

## Operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
z = torch.add(x, y)
y.add_(x) # _ means inplace
print(y)
print(z)

z = x - y
z = torch.sub(x, y)
print(z)

z = x * y
z = torch.mul(x, y)
print(z)

z = x / y
z = torch.div(x, y)
print(z)

x = torch.tensor([2, 2])
y = torch.tensor([3, 2])
z = x ** y # element-wise power
print(z)

x = torch.rand(5, 3)
print(x)

print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1, 1]) # element at row 1, column 1


## Reshape
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8) # -1 means infer from other dimensions
print(z)
print(x.size(), y.size(), z.size())

## Convert to numpy
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy() # b is a numpy array
print(b)
c = a + b  # c is a torch tensor because a is a torch tensor and b is a numpy array
print(c)

print(type(a))
print(type(b))
print(type(c))

print(a)
a += 1
print(a)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda") # a cuda device object
    x = torch.ones(5, device=device) # create a tensor on GPU
    y = torch.ones(5) # create a tensor on CPU
    y = y.to(device) # move y to GPU
    z = x + y # add two tensors on GPU
    z = z.to("cpu") # move z to CPU
    print(z)


## Create a decorator timer
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

@timer
def test():
    time.sleep(1)

test()

@timer
def gpuMultiply():
    device = torch.device("cuda")
    x = torch.rand(10000, 10000, device=device)
    y = torch.rand(10000, 10000, device=device)
    z = torch.matmul(x, y)

@timer
def cpuMultiply():
    x = torch.rand(10000, 10000)
    y = torch.rand(10000, 10000)
    z = torch.matmul(x, y)

gpuMultiply()
cpuMultiply()

## Autograd
x = torch.ones(5, requires_grad=True) # is needed if we want to compute gradients
print(x)

x = torch.randn(3, requires_grad=True) # will create a computational graph
print(x)

'''
Computational graph:

Forward pass ->
x ->
    ⊕ -> y           ---
2 ->                   |     
<- Add Backward pass <--

'''

y = x + 2
print(y)

z = y * y * 2
z = z.mean()
print(z)

z.backward() # dz/dx
print(x.grad)

####
x = torch.randn(3, requires_grad=True)
y = x + 2
z = y * y * 2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32) # v is a jacobian vector?
z.backward(v) # dz/dx
print(x.grad)

## Disable gradient tracking
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
#     y = x + 2
#     print(y)

x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)

x = torch.randn(3, requires_grad=True)
x.detach_()

x = torch.randn(3, requires_grad=True)
y = None
with torch.no_grad():
    y = x + 2
    print(y)


##
weights = torch.ones(4, requires_grad=True)
for epoch in range(9):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # zero out the gradients

## Optimizer
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

# 04 BACKPROPAGATION
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
def forward(x, w): 
    def y_hat(w, x): # y_hat is the prediction
        return w * x

    def s(y_hat, y): 
        return y_hat - y

    def loss(s): 
        return s ** 2

    return loss(s(y_hat(w, x), y))

loss = forward(x, w)
print(loss)
loss.backward()
print(w.grad)

# 05 Gradient Descent with Autograd and Backpropagation
import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

# model prediction
def forward(x): 
    return w * x

# loss
def loss(y, y_predicted): 
    return ((y_predicted - y) ** 2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y) # chain rule
def gradient(x, y, y_predicted): 
    return np.dot(2 * x, y_predicted - y).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")


# let's do it with PyTorch
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x): 
    return w * x

# loss
def loss(y, y_predicted): 
    return ((y_predicted - y) ** 2).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")

# 06 Training Pipeline: Model, Loss, and Optimizer
import torch

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

# Training
learning_rate = 0.10
n_iters = 100

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")



# 07 Linear Regression
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
# Linear model f = wx + b
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss() # mean squared error - loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # stochastic gradient descent - optimizer

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()


# 08 Logistic Regression
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# cast to float Tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
# reshape
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.lin(x))

model = LogisticRegression(n_features)

# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.BCELoss() # binary cross entropy - loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # stochastic gradient descent - optimizer

# 3) Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Evaluate
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc:.4f}')

