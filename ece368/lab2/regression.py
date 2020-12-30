import numpy as np
import matplotlib.pyplot as plt
import util

# Use a flag as a quick option that will be disabled on submit because some people may not have seaborn
USE_SEABORN = True
if USE_SEABORN:
    import seaborn as sns

    sns.set()
    sns.set_context('paper')

# Graphing parameters
NUM_PTS = 100
TRUE_A = [-0.1, -0.5]
LEVELS = 3
X1, X2 = np.meshgrid(np.linspace(-1.0, 1.0, NUM_PTS), np.linspace(-1.0, 1.0, NUM_PTS))


def draw_contour_plot(mean, cov, num_points):
    """
    Call to create contour plot with correct axis labels and limits and plotting the true value of a
    """
    # hard code a case for the prior
    if num_points == 0:
        title = 'Prior Distribution'
        save_name = 'prior.pdf'
    else:
        title = 'Posterior Distribution with ' + str(num_points) + ' point(s)'
        save_name = 'posterior' + str(num_points) + '.pdf'

    vals = np.zeros(X1.shape)
    for i in range(NUM_PTS):
        for j in range(NUM_PTS):
            vals[i][j] = util.density_Gaussian(mean, cov, np.array([[X1[i][j], X2[i][j]]]))
    c = plt.contour(X1, X2, vals, LEVELS)
    plt.clabel(c)

    c = plt.scatter(x=TRUE_A[0], y=TRUE_A[1], marker='x')
    c.set_label('True a')
    plt.legend()
    plt.title(title)
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(save_name)
    plt.show()
    return


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    mean = np.array([0, 0])
    cov = np.array([[beta, 0], [0, beta]])
    draw_contour_plot(mean, cov, 0)
    return


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    # x is 1D so X transpose times x is a scalar
    reg = sigma2 / beta
    new_x = np.ones((x.size, 2))
    for idx, val in enumerate(x):
        new_x[idx][1] = val
    s1 = np.dot(np.transpose(new_x), new_x)
    s1 = s1 + reg * np.identity(s1.shape[0])
    s2 = np.dot(np.linalg.inv(s1), np.transpose(new_x))
    mu = np.dot(s2, z)
    mu = np.reshape(mu, (2,))
    Cov = sigma2 * np.linalg.inv(s1)
    draw_contour_plot(mu, Cov, len(z))
    return (mu, Cov)


def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    # Perform multiplication manually - this is equivalent to a1 * xi + a0 for all xi in x
    z_preds = np.array([mu[1] * e + mu[0] for e in x])

    # For the covariance calculation, properly construct the X matrix with entries [1 xi]
    x_vals = np.array([[1, e] for e in x])
    z_pred_cov = np.dot(x_vals, np.dot(Cov, np.transpose(x_vals))) + sigma2
    z_pred_stddev = np.sqrt(np.diag(z_pred_cov))

    c = plt.errorbar(x=x, y=z_preds, yerr=z_pred_stddev, uplims=True, lolims=True)
    c.set_label('Predictions')
    c = plt.scatter(x=x_train, y=z_train, marker='.')
    c.set_label('Training Points')
    plt.legend()
    plt.title('Prediction Graph using ' + str(len(z_train)) + ' training point(s)')
    plt.xlabel('Input')
    plt.ylabel('Target')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig('predict' + str(x_train.shape[0]) + '.pdf')
    plt.show()
    return


if __name__ == '__main__':
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]

    # known parameters 
    sigma2 = 0.1
    beta = 1

    # number of training samples used to compute posterior
    # ns = 5

    # used samples
    # x = x_train[0:ns]
    # z = z_train[0:ns]

    # prior distribution p(a)
    priorDistribution(beta)

    # posterior distribution p(a|x,z)
    for i in [1, 5, 100]:
        x = x_train[0:i]
        z = z_train[0:i]
        mu, Cov = posteriorDistribution(x, z, beta, sigma2)

        # distribution of the prediction
        predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)
