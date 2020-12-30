import os.path
import numpy as np
import matplotlib.pyplot as plt
import util


class SmoothingCounter(dict):
    """
    A smoothed version of the Counter class.
    This should not be used outside of initialization with initialize_from_counter.
    """

    def __init__(self):
        super(SmoothingCounter, self).__init__()
        self.size = 0

    def initialize_from_counter(self, counter_obj: util.Counter, total_words: int, word_count: int):
        ct = 0
        for k, v in counter_obj.items():
            ct += v
        self.size = total_words
        for k, v in counter_obj.items():
            self[k] = (v + 1) / (ct + word_count)
        return

    def __missing__(self, key):
        return 1 / self.size


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    probabilities_by_category = (SmoothingCounter(), SmoothingCounter())

    spam_counter = util.get_word_freq(file_lists_by_category[0])
    ham_counter = util.get_word_freq(file_lists_by_category[1])

    # Brute force the intersect
    intersect = 0
    for k in ham_counter:
        if k in spam_counter:
            intersect += 1
    union_count = len(spam_counter) + len(ham_counter) - intersect  # length of the full word list

    full_count = 0
    for k, v in ham_counter.items():
        full_count += v
    for k, v in spam_counter.items():
        full_count += v

    # Add spam
    probabilities_by_category[0].initialize_from_counter(spam_counter, full_count, union_count)

    # Add ham
    probabilities_by_category[1].initialize_from_counter(ham_counter, full_count, union_count)

    return probabilities_by_category


def classify_new_email(filename, probabilities_by_category, prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    words = util.get_word_freq([filename])
    spam_prob, ham_prob = 0.0, 0.0

    for k, v in words.items():
        spam_prob += v * np.log(probabilities_by_category[0][k])
        ham_prob += v * np.log(probabilities_by_category[1][k])

    spam_prob *= prior_by_category[0]
    ham_prob *= prior_by_category[1]

    res_string = 'spam' if spam_prob > ham_prob else 'ham'

    # This won't affect the comparison but for the correct log probabilities we need to add the denominator
    spam_prob, ham_prob = spam_prob / (spam_prob + ham_prob), ham_prob / (spam_prob + ham_prob)

    classify_result = (res_string, [spam_prob, ham_prob])

    return classify_result


if __name__ == '__main__':

    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2, 2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label, log_posterior = classify_new_email(filename,
                                                  probabilities_by_category,
                                                  priors_by_category)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0], totals[0], correct[1], totals[1]))

    print('Type 1 Errors:', performance_measures[0][1])  # assuming hypothesis to be 'This email is spam.'
    print('Type 2 Errors:', performance_measures[1][0])

    error_coeff = np.append([0.0], np.linspace(0.9, 1.1, 8))  # add a really small value for zero Type 1 errors
    error_coeff = np.append(error_coeff, [1000.0])  # add a really large value for zero Type 2 errors
    t1_err, t2_err = np.zeros(10), np.zeros(10)
    files = (util.get_files_in_folder(test_folder))
    for idx, val in enumerate(error_coeff):
        performance_measures = np.zeros([2, 2])
        for filename in files:
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            # Replace the decision rule with my own based on the log probabilities
            guessed_index = (log_posterior[0] / log_posterior[1]) > val
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "Correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        correct = np.diag(performance_measures)
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        t1_err[idx] = performance_measures[0][1]
        t2_err[idx] = performance_measures[1][0]

    # Use seaborn so it's prettier (but it may not exist on all computers so set a flag)
    USE_SEABORN = False
    if USE_SEABORN:
        import seaborn as sns
        sns.set()
        sns.set_context('paper')

    plt.plot(t1_err, t2_err, '.')
    plt.title('Type 1 Error vs Type 2 Error')
    plt.xlabel('Type 1 Errors')
    plt.ylabel('Type 2 Errors')
    plt.show()
