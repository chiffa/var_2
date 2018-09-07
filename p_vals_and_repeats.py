import numpy as np
from scipy.stats import t, norm, ttest_ind
from matplotlib import pyplot as plt
from os import urandom

def generate_data(mean_1, mean_2, samples):
    return norm.rvs(loc=mean_1, size=samples), norm.rvs(loc=mean_2, size=samples)


def test_p_val_distribution(mean_1, mean_2, samples, tests):
    p_val_list = []
    for _ in range(0, tests):
        _, p_val = ttest_ind(*generate_data_fixed(mean_1, mean_2, samples))
        p_val_list.append(p_val)

    plt.hist(p_val_list)
    plt.show()


def generate_data_fixed(mean_1, mean_2, samples):
    return norm.rvs(loc=mean_1, size=samples, random_state=42), \
           norm.rvs(loc=mean_2, size=samples, random_state=66)


if __name__ == "__main__":
    test_p_val_distribution(1, 1, 5, 500)
    test_p_val_distribution(1, 2, 5, 500)

