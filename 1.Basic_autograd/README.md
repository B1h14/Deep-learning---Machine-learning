This this a basic autograd implementation for scalar functions.
This code is a solution to the following problem : https://www.deep-ml.com/problems/26
This solution works for computation chains that represent an acyclic graph. Instead of topological sorting we use a natural way of sorting that uses the linearity of the operations used at the cost of more memory usage.

The file basic_autograd.py contains the class Value.
An object with class Value contains its gradient , value and the definition of some basic operations : sum and multiplication of two elements , relu function .

the test_autograd.py file contains a test case for the code provided in the original problem site.