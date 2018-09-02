import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

def generate(x0, A, B, C, D, num_obs = 100):
    '''creates sequences of true state (x)
       and observations (y)'''
    x = [x0] + [None] * (num_obs - 1)
    y = [None] * (num_obs)

    for k in range(num_obs - 1):
        x[k+1] = A + B * x[k] + C * np.random.normal()
        y[k] = x[k] + D * np.random.normal()

    y[k+1] = x[k+1] + D * np.random.normal()
    return x, y

def kalman_filter(y, A, B, C, D, initial = 'default'):
    '''takes in a list of observations y and
        estimates for A, B, C, D as outlined in
        Elliot (2005) and returns estimate of 
        the true state using Kalman filters.
        
        initial is a tuple of the initial values of
        the estimate of the state and it's variance
        in period zero.'''
    
    if initial == 'default':
        initial = (y[0], D ** 2)
    
    N = len(y) - 1 
    x = [0] * (N + 1) # state estimations
    R = [0] * (N + 1) # variance estimations sigma k | k
    sigma = [None] * (N + 1) # sigma k + 1 | k
    
    #initializing x and R
    x[0], R[0] = initial

    for k in range(N):
        # Equation 22
        x_bar = A + B * x[k]

        # Equation 23
        sigma[k+1] = B ** 2 * R[k] + C ** 2

        # Equation 24
        K = sigma[k+1] / (sigma[k+1] + D ** 2)

        # Equation 25
        x[k+1] = x_bar + K * (y[k+1] - x_bar)

        # Equation 26
        R[k+1] = sigma[k+1] * (1 - K)
    
    return x, R, sigma, K

def EM(y, max_iterations = 100, threshold = 0.05,
        initial_eta = (0.5, 0.5, 0.5, 0.5),
        initial_kalman = (0, 0.1)):
    '''Uses EM Algorithm for estimating A, B, C, D given
        observations of data.
        
        To decode from Elliot(2005):
            - x[k]       == \hat{x}_{k|k}
            - x_N[k]     == \hat{x}_{k|N}
            - R[k]       == \Sigma_{k|k}
            - sigma[k]   == \Sigma_{k+1|k}
            - sigma_N[k] == \Sigma_{k|N}
            - cov_N[k]   == \Sigma_{k-1,k|N}
            - J[k]       == J_k
            - K          == K_N
    '''
    
    #initialize A, B, C, D
    A, B, C, D = initial_eta
    eta = [(A, B, C, D)] #list of parameters at each step

    N = len(y) - 1
    x_N = [None] * (N + 1)      #31
    sigma_N = [None] * (N + 1)  #32
    cov_N = [None] * (N + 1)    #33
    J = [None] * (N + 1)

    #initalize number of iterations and done flag
    i, done = 0, False

    while not done:
        #compute kalman filter with current estimates
        x, R, sigma, K = kalman_filter(y, A, B, C, D, initial_kalman)

        #initialize with results from kalman filter
        x_N[N] = x[N]
        sigma_N[N] = sigma[N]

        #calculating smoothers using backwards recursion
        for k in reversed(range(N)):
            # Equation 34
            J[k] = B * R[k] / sigma[k+1]

            # Equation 35
            x_N[k] = x[k] + J[k] * (x_N[k+1] - (A + B * x[k]))

            # Equation 36
            sigma_N[k] = R[k] + J[k]**2 * (sigma_N[k+1] - sigma[k+1])
        
        #cov uses second index
        #i.e. Sigma_k-1,k|N == cov_N[k]
        # Equation 38
        cov_N[N] = B * (1 - K) * R[N-1]

        for k in reversed(range(1, N)):
            # Equation 37
            val = J[k-1] * R[k]
            val += J[k] * J[k-1] * (cov_N[k+1] - B * R[k])
            cov_N[k] = val

        #helper values
        x_N_2 = [k ** 2 for k in x_N]
        alpha = sum(sigma_N[0:N]) + sum(x_N_2[0:N])
        beta = sum(cov_N[1:]) + sum([a * b for a, b in zip(x[0:N], x[1:])])
        gamma = sum(x_N[1:])
        delta = gamma - x_N[N] + x_N[0]

        #calculating new estimates for A, B, C, D
        A_hat = (alpha * gamma - delta * beta) / (N * alpha - delta ** 2)
        B_hat = (N * beta - gamma * delta) / (N * alpha - delta ** 2)

        C_hat, D_hat = 0, 0
        for k in range(1, N + 1):
            C_hat += sigma_N[k] + x_N[k] ** 2 + A_hat ** 2
            C_hat += B_hat ** 2 * (sigma_N[k-1] + x_N[k-1] ** 2)
            C_hat -= 2 * A_hat * x_N[k]
            C_hat += 2 * A_hat * B_hat * x_N[k-1]
            C_hat -= 2 * B_hat * cov_N[k]
            C_hat -= 2 * B_hat * x_N[k] * x_N[k-1]

        C_hat = C_hat / N
        C_hat = sqrt(C_hat)

        D_hat = sum([a ** 2 for a in y]) - sum([2 * a * b for a, b in zip(y, x_N)])
        D_hat += sum(sigma_N) + sum(x_N_2)
        D_hat = D_hat / (N + 1)
        D_hat = sqrt(D_hat)
        
        #changing initial kalman estimates for the next iteration
        initial_kalman = (x_N[0], sigma_N[0])
        #initial_kalman = (x_N[0], 0.1)
        
        #checking done conditions
        dist = (A - A_hat) ** 2 + (B - B_hat) ** 2
        dist += (C - C_hat) ** 2 + (D - D_hat) ** 2
        i += 1
        if i == max_iterations or dist < threshold:
            done = True

        A, B, C, D = A_hat, B_hat, C_hat, D_hat
        eta.append((A, B, C, D))
    return A, B, C, D, eta, x_N

def plot_eta(eta, path = None):
    A_lst, B_lst, C_lst, D_lst = zip(*eta)
    fig, axes = plt.subplots(4, figsize = (8, 15))

    axes[0].plot(A_lst);
    axes[0].set_title('A');
    axes[1].plot(B_lst);
    axes[1].set_title('B');
    axes[2].plot(C_lst);
    axes[2].set_title('C');
    axes[3].plot(D_lst);
    axes[3].set_title('D');

    plt.show()

    fig.savefig(path)