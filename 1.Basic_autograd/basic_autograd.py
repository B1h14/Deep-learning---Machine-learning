# Basic Autograd Implementation for scalar values
class Value:
	def __init__(self, data, _children=[], _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda x : None  # Placeholder for backward function
		self._prev = _children
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		result = Value(self.data + other.data, [self, other], '+')
		def fun(result):
			self.grad +=  result.grad
			other.grad +=  result.grad
			self._backward(self)
			other._backward(other)
			
		result._backward = fun
		return result

	def __mul__(self, other):
		result = Value(self.data * other.data, [self, other], '*')
		def fun(result):
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad
			self._backward(self)
			other._backward(other)
		result._backward = fun
		return result

	def relu(self):
		# Implement ReLU here
		result = Value(max(0, self.data), [self], 'relu')
		def fun(result):
			self.grad += (result.data > 0) * result.grad
			self._backward(self)
		result._backward = fun
		return result

	def backward(self):
		# Implement backward pass here
		self.grad = 1
		self._backward(self)