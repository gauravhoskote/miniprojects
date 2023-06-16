import numpy as np
import pandas as pd
import torch

X = torch.tensor([1,2,3,4], dtype= torch.float32)
Y = torch.tensor([1,2,3,4], dtype= torch.float32)
w = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)

def forward(x):
  return w * x

def loss(y, ypred):
  return ((ypred - y)**2).mean()

print(f'Prediction before training: f(3):{forward(3):.3f}')

learning_rate = 0.01
n_iter = 1000

for i in range(n_iter):
  ypred = forward(X)
  l = loss(Y, ypred)
  l.backward()
  with torch.no_grad():
    w.copy_(w - (learning_rate * w.grad))

  w.grad.zero_()
  if i % 10 == 0:
    print(f'epoch : {i+1} loss : {l} w: {w}')

