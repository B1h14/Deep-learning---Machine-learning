import math

def gamma(k):
    """
    Calculate the gamma function value for a given k.
    """
    # Direct calculation with math library of the gamma value
    #return math.gamma(k/2)
    # Custom implementation using gamma properties and the value of gamma at 1/2
    if k == 1:
        return 1
    elif k == 1/2:
        return math.sqrt(math.pi)
    else:
        return (k - 1) * gamma(k - 1)

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    probability = (1 / (2 ** (k / 2) * gamma(k / 2))) * (x ** (k / 2 - 1)) * math.exp(-x / 2)
    return round(probability, 3)