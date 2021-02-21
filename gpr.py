import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# and some stats
# def mean(x):

'''train_set = pd.read_csv('C:/Users/ASUS/Documents/input/train.csv')
test_set = pd.read_csv('C:/Users/ASUS/Documents/input/test.csv')

train_set = train_set.dropna()
test_set = test_set.dropna()

# print("Rows before clean: ", dirty_training_set.size, "\n")
print("Rows after clean: ", train_set.size, "\n")
print("Rows after clean: ", test_set.size, "\n")

x_train = train_set['x'].to_numpy()
y_train = train_set['y'].to_numpy()

x_test = test_set['x'].to_numpy()
y_test = test_set['y'].to_numpy()'''

x_train = np.array([-9, -5, 3, 6, 8]).reshape(-1, 1)
y_train = np.array([-8, -7, -2, 1, 9]).reshape(-1, 1)

x_test = np.array(
    [-10.0000000, -9.591836, -9.1836735, -8.7755102, -8.3673469, -7.9591837, -7.5510204, -7.1428571, -6.7346939,
     -6.3265306, -5.9183673, -5.5102041, -5.1020408, -4.6938776, -4.2857143, -3.8775510, -3.4693878, -3.0612245,
     -2.6530612, -2.2448980, -1.8367347, -1.4285714, -1.0204082, -0.6122449, -0.2040816, 0.2040816, 0.6122449,
     1.0204082, 1.4285714, 1.8367347, 2.2448980, 2.6530612, 3.0612245, 3.4693878, 3.8775510, 4.2857143, 4.6938776,
     5.1020408, 5.5102041, 5.9183673, 6.3265306, 6.7346939, 7.1428571, 7.5510204, 7.9591837, 8.3673469, 8.7755102,
     9.1836735, 9.5918367, 10.0000000]).reshape(-1, 1)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def posterior(x_test, x_train, y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(x_train, x_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(x_train))
    K_s = kernel(x_train, x_test, l, sigma_f)
    K_ss = kernel(x_test, x_test, l, sigma_f) + 1e-8 * np.eye(len(x_test))
    K_s_T = K_s.transpose()

    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s_T.dot(K_inv).dot(y_train)

    # Equation (8)
    cov_s = K_ss - K_s_T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(x_test, x_train, y_train)
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 10)
plot_gp(mu_s, cov_s, x_test, X_train=x_train, Y_train=y_train, samples=samples)
