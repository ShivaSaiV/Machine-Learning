import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def get_data(mean: float, std: float, size: int) -> np.ndarray:
    """
    Generate data from a Gaussian distribution.

    >>> np.random.seed(42)
    >>> data = get_data(0, 1, 1000)
    >>> len(data)
    1000
    >>> bool(-1 < np.mean(data) < 1)
    True
    """
    # >>> YOUR CODE HERE >>>
    data = np.random.normal(mean, std, size)
    # <<< END OF YOUR CODE <<<
    return data

def log_pdf(data: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """
    Calculate the log probability density function for Gaussian distribution.

    >>> data = np.array([0, 1, 2])
    >>> np.round(log_pdf(data, 1, 1), 4)
    array([-1.4189, -0.9189, -1.4189])
    """
    # >>> YOUR CODE HERE >>>
    pdf = (1.0 / np.sqrt(2 * np.pi * (variance))) * (np.e ** (-((data - mean) ** 2) / (2 * variance)))
    log_p = np.log(pdf)
    # <<< END OF YOUR CODE <<<
    return log_p

def histogram(data: np.ndarray):
    """
    Plot a histogram of the data with a vertical line at the mean.

    >>> np.random.seed(42)
    >>> data = np.random.normal(0, 1, 1000)
    >>> histogram(data)  # This will save a plot
    """
    # >>> YOUR CODE HERE >>>
    histogram = plt.hist(data, color = "blue")
    plt.axvline(np.mean(data), color = "black")
    plt.title("Histogram")
    plt.savefig("histogram.png")
    plt.close()
    # <<< END OF YOUR CODE <<<

def mle(data: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Maximum Likelihood Estimates for mean and variance.

    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mu, var = mle(data)
    >>> round(mu, 4), round(var, 4)
    (3.0, 2.0)
    """
    # >>> YOUR CODE HERE >>>
    mu = np.mean(data)
    sigma_squared = np.mean((data - mu) ** 2)
    # <<< END OF YOUR CODE <<<
    return mu, sigma_squared

def gradient(data: np.ndarray, mean: float, variance: float) -> Tuple[float, float]:
    """
    Calculate the gradient of the negative log-likelihood.

    >>> data = np.array([2, 2, 3])
    >>> grad_mu, grad_var = gradient(data, 2, 1)
    >>> round(grad_mu, 4), round(grad_var, 4)
    (-1.0, 1.0)
    """
    # >>> YOUR CODE HERE >>>

    sumElements = 0
    for i in data:
        sumElements += (i - mean)
    grad_mu = -1 * (1.0 / variance) * sumElements

    sumSquared = 0
    for i in data:
        sumSquared += ((i - mean) ** 2)
    grad_variance = -1 * (sumSquared / (2 * (variance ** 2))) + (len(data) / 2 * variance)

    # <<< END OF YOUR CODE <<<
    return grad_mu, grad_variance

def fit(data: np.ndarray, lr: float = 0.001, n_iterations: int = 100, tolerance: float = 1e-10) -> Tuple[float, float]:
    """
    Use Gradient Descent to find maximum likelihood estimates for mean and variance.

    >>> np.random.seed(42)
    >>> data = np.random.normal(2, 1.5, 1000)
    >>> mu, var = fit(data)
    >>> 1.9 < mu < 2.1 and 2.0 < var < 2.5
    True
    """
    mu, sigma_squared = mle(data)  # Initialize with MLE estimates
    for _ in range(n_iterations):
        # >>> YOUR CODE HERE >>>
        gd_mean, gd_var = gradient(data, mu, sigma_squared)
        next_mu = mu - lr * gd_mean
        next_var = sigma_squared - lr * gd_var
        if (abs(mu - next_mu) < tolerance):
            break
        else:
            mu, sigma_squared = next_mu, next_var
        # <<< END OF YOUR CODE <<<
    return mu, sigma_squared

if __name__ == "__main__":
    np.random.seed(42)
    import doctest
    doctest.testmod()
    
    true_mean, true_std = 2.0, 3.0
    sample_size = 1000
    data = get_data(true_mean, true_std, sample_size)

    histogram(data)

    mle_mean, mle_var = mle(data)
    print(f"MLE Estimates - Mean: {mle_mean:.4f}, Variance: {mle_var:.4f}")

    gd_mean, gd_var = fit(data)
    print(f"Gradient Descent Estimates - Mean: {gd_mean:.4f}, Variance: {gd_var:.4f}")
