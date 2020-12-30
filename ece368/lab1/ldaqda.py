import numpy as np
import matplotlib.pyplot as plt
import util


def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    # Create lists with only male and female data
    n = len(y)
    male_arr, female_arr = [], []
    for i in range(n):
        if y[i] == 1:
            male_arr.append(x[i])
        elif y[i] == 2:
            female_arr.append(x[i])
        else:
            print('Incorrect label')
    male_samples = len(male_arr)
    female_samples = len(female_arr)
    male_arr, female_arr = np.array(male_arr), np.array(female_arr)

    # Get means
    mu_male = np.mean(male_arr, axis=0)
    mu_female = np.mean(female_arr, axis=0)
    total_mean = (np.sum(male_arr) + np.sum(female_arr)) / n

    # Get covs
    cov_male, cov_female, cov = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    for i in range(male_samples):
        t = male_arr[i] - mu_male
        s = t.reshape(2, 1)
        cov_male += np.dot(s, s.T)
    for i in range(female_samples):
        t = female_arr[i] - mu_female
        s = t.reshape(2, 1)
        cov_female += np.dot(s, s.T)
    for i in range(n):
        t = np.array(x[i] - total_mean)
        s = t.reshape(2, 1)
        cov += np.dot(s, s.T)

    cov_male /= male_samples
    cov_female /= female_samples
    cov /= n

    return mu_male, mu_female, cov, cov_male, cov_female


def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    n = len(y)

    male_probs_lda = util.density_Gaussian(mu_male, cov, x)
    female_probs_lda = util.density_Gaussian(mu_female, cov, x)
    lda_preds = [1 if male_probs_lda[i] >= female_probs_lda[i] else 2 for i in range(n)]

    male_probs_qda = util.density_Gaussian(mu_male, cov_male, x)
    female_probs_qda = util.density_Gaussian(mu_female, cov_female, x)
    qda_preds = [1 if male_probs_qda[i] >= female_probs_qda[i] else 2 for i in range(n)]

    mis_lda = np.sum(np.abs(np.array(lda_preds) - np.array(y))) / n  # abs(y_hat-y) = 0 if correct pred else 1
    mis_qda = np.sum(np.abs(np.array(qda_preds) - np.array(y))) / n

    m1, m2, f1, f2 = [], [], [], []
    for idx, val in enumerate(x):
        if lda_preds[idx] == 1:
            m1.append(val[0])
            m2.append(val[1])
        else:
            f1.append(val[0])
            f2.append(val[1])

    NUM_PTS = 1000
    # Use seaborn so it's prettier (but it may not exist on all computers so set a flag)
    USE_SEABORN = False
    if USE_SEABORN:
        import seaborn as sns
        sns.set()
        sns.set_context('paper')

    x1, x2 = np.meshgrid(np.linspace(54, 80, NUM_PTS), np.linspace(100, 280, NUM_PTS))
    vals1, vals2, preds = np.zeros(x1.shape), np.zeros(x1.shape), np.zeros(x1.shape)
    for i in range(NUM_PTS):
        for j in range(NUM_PTS):
            v = np.array([x1[i][j], x2[i][j]])
            v = v.reshape((1,2))
            vals1[i][j] = util.density_Gaussian(mu_male, cov, v)
            vals2[i][j] = util.density_Gaussian(mu_female, cov, v)
            preds[i][j] = vals1[i][j] > vals2[i][j]
    plt.contour(x1, x2, preds, 1)
    c = plt.scatter(x=m1, y=m2, marker='.')
    c.set_label('Male')
    c = plt.scatter(x=f1, y=f2, marker='x')
    c.set_label('Female')
    c = plt.scatter(x=mu_male[0], y=mu_male[1], marker='*')
    c.set_label('Male Mean')
    c = plt.scatter(x=mu_female[0], y=mu_female[1], marker='P')
    c.set_label('Female Mean')
    plt.legend()
    plt.title('LDA Plot (Solid Line is Decision Boundary)')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.show()

    m1, m2, f1, f2 = [], [], [], []
    for idx, val in enumerate(x):
        if qda_preds[idx] == 1:
            m1.append(val[0])
            m2.append(val[1])
        else:
            f1.append(val[0])
            f2.append(val[1])
    for i in range(NUM_PTS):
        for j in range(NUM_PTS):
            v = np.array([x1[i][j], x2[i][j]])
            v = v.reshape((1,2))
            vals1[i][j] = util.density_Gaussian(mu_male, cov_male, v)
            vals2[i][j] = util.density_Gaussian(mu_female, cov_female, v)
            preds[i][j] = vals1[i][j] > vals2[i][j]
    plt.contour(x1, x2, vals1, 5, linestyles='dashed')
    plt.contour(x1, x2, vals2, 5, linestyles='dashed')
    plt.contour(x1, x2, preds, 1)
    c = plt.scatter(x=m1, y=m2, marker='.')
    c.set_label('Male')
    c = plt.scatter(x=f1, y=f2, marker='x')
    c.set_label('Female')
    c = plt.scatter(x=mu_male[0], y=mu_male[1], marker='*')
    c.set_label('Male Mean')
    c = plt.scatter(x=mu_female[0], y=mu_female[1], marker='P')
    c.set_label('Female Mean')
    plt.legend()
    plt.title('QDA Plot (Solid Line is Decision Boundary)')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.show()

    return mis_lda, mis_qda


if __name__ == '__main__':
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train, y_train)

    # misclassification rate computation
    mis_LDA, mis_QDA = misRate(mu_male, mu_female, cov, cov_male, cov_female, x_test, y_test)
    print('LDA Misclassification Rate:', mis_LDA, 'QDA Misclassification Rate:', mis_QDA)

    # mu = (mu_male + mu_female) / 2
    # sigma_inv = np.linalg.inv(cov)
    # beta = np.dot(sigma_inv, mu)
    # gamma = np.log(0.5) - 0.5 * np.dot(mu, np.dot(sigma_inv, mu.T))
