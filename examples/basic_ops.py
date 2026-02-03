import sys
sys.path.insert(0, '/home/claude')
import minitorch as mt

x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True)
y = mt.Tensor([[5, 6], [7, 8]], requires_grad=True)

print("x:", x)
print("y:", y)

z = x + y
print("x + y:", z)

w = x @ y
print("x @ y:", w)

loss = w.sum()
print("sum(x @ y):", loss)

loss.backward()
print("x.grad:", x.grad)
print("y.grad:", y.grad)

print("\nTesting activation functions:")
a = mt.Tensor([[-1, 0, 1, 2]], requires_grad=True)
print("Input:", a)
print("ReLU:", a.relu())
print("Sigmoid:", a.sigmoid())
print("Tanh:", a.tanh())

print("\nTesting reshape and transpose:")
b = mt.Tensor([[1, 2, 3], [4, 5, 6]])
print("Original:", b)
print("Reshaped to (3, 2):", b.reshape(3, 2))
print("Transposed:", b.transpose())

print("\nTesting reduction operations:")
c = mt.Tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor:", c)
print("Sum:", c.sum())
print("Mean:", c.mean())
print("Max:", c.max())
print("Sum along dim=0:", c.sum(dim=0))
print("Sum along dim=1:", c.sum(dim=1))
