import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


class Value:
  def __init__(self,data,_children=(),_op='',label=''):
    self.grad = 0.0
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self._backward = lambda: None
  def __repr__(self):
    return f"value(data={self.data})"
  def __add__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data + other.data,(self,other),'+')
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __sub__(self,other):
    return self + (-other)

  def __rmul__(self, other):
    return self * other

  def __mul__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data * other.data,(self,other),'*')
    def _backward():
      self.grad += (other.data) * out.grad
      other.grad += (self.data) * out.grad
    out._backward = _backward
    return out
  def exp(self):
    x = self.data
    out = Value(math.exp(x),(self,),'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self,other):
    assert isinstance(other,(int,float)), "only supporting int/float powers for now"
    out = Value(self.data**other,(self,),f'**{other}')
    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out

  def __truediv__(self,other):
    return self * other**-1

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t,(self,),'tanh')
    def _backward():
      self.grad += (1-t**2) * out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()



x1 = Value(2.0,label='x1')
x2 = Value(0.0,label='x2')
w1 = Value(-3.0,label='w1')
w2 = Value(1.0,label='w2')
b  = Value(6.88845,label='b')
x1w1 = x1*w1; x1w1.label = 'x1*w1';
x2w2 = x2*w2; x2w2.label = 'x2*w2';
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2';
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh()
o.label = 'o'
o.backward()
draw_dot(o)
# a = Value(2.0)
# b = Value(4.0)
# a / b


import random
class Neuron:
  def __init__(self,nin):
    self.w = [ Value(random.uniform(-1,1)) for _ in range(nin) ]
    self.b = Value(random.uniform(-1,1))

  def __call__(self,x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)),self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b];

class Layer:
  def __init__(self,nin,nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    params = []
    for neuron in self.neurons:
      ps = neuron.parameters()
      params.extend(ps)
    return params

class MLP:
  def __init__(self,nin,nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

x = [2.0,3.0,-1.0]

n = MLP(3,[4,4,1])
n(x)



xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

for k in range(20):
  #forward pass
  ypred = [n(x) for x in xs]
  loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)], start=Value(0.0))

  #backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  #update
  for p in n.parameters():
    p.data += -0.05 * p.grad

  print(k,loss.data)

