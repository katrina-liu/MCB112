"""
Helper functions for section 6 notebook
Danylo Lavrentovich 2020
Modified by Aoyue Mao (2021)
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# I like bigger fonts than the default matplotlib ones...
def set_font_sizes(SMALL_SIZE=14, MEDIUM_SIZE=16, LARGE_SIZE=20):
    '''
    Sets custom font size for matplotlib
    From: https://stackoverflow.com/a/39566040
    '''
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title
    
    
def viz_coin_flip_H0_H1(k, n, p_null, show_h1=False):
    '''
    Given:
        a particular number of heads, k
        a number of coin flips, n
        a probability of heads, p_null
        an optional parameter to show an alternative hypothesis also
            based on the max likelihood estimate of p
    Plots the PDF and a CDF of a binomial process given the null hypothesis, 
        highlighting the particular number of heads, k,
    '''    
    # the possible number of heads is 0 to n
    ks = np.arange(n+1)
    
    # the max likelihood estimate of p given data, k/n
    p_h1 = k/n
    
    # plot PMF
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    axs[0].plot(ks, stats.binom.pmf(ks, n, p_null),'b.-', label=r'$H_0: p = {:.2f}$'.format(p_null))
    if show_h1:
        axs[0].plot(ks, stats.binom.pmf(ks, n, p_h1),'r.-',  label=r'$H_1: p = {:.2f}$'.format(p_h1))
    axs[0].set_xlabel('$k$ heads out of n={} flips'.format(n))
    axs[0].set_ylabel(r'$P(k$ heads $\mid H_0: p = 0.5)$')
    axs[0].set_title('probability mass function, PMF')
    axs[0].axvline(k, color='k', ls='--', label='observed data')

    # plot CDF
    axs[1].plot(ks, stats.binom.cdf(ks, n, p_null),'b.-',label=r'$H_0: p = {:.2f}$'.format(p_null))
    if show_h1:
        axs[1].plot(ks, stats.binom.cdf(ks, n, p_h1),'r.-', label=r'$H_1: p = {:.2f}$'.format(p_h1))
    axs[1].set_xlabel('$k$ heads out of n={} flips'.format(n))
    axs[1].set_ylabel(r'$P($up to $k$ heads $\mid H_0: p = 0.5)$')
    axs[1].set_title('cumulative distribution function, CDF')
    axs[1].axvline(k, color='k', ls='--', label='observed data')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.subplots_adjust(wspace=0.35)
    plt.show()
        
    
def estimate_normal_from_sample(n, true_mu, true_sigma, seed=-1):
    '''
    Given:
        a number of data points to sample, n
        a true mean of a Gaussian, true_mu
        a true standard deviation of a Gaussian, true_sigma
        an optional random seed, seed
    Generates a sample of n from a Gaussian with supplied parameters
    Computes sample mean (xbar) and sample standard deviation (S)
    Plots a histogram of the sample data and an "estimated" Gaussian using xbar and S.
    '''
    if seed < 0:
        seed = np.random.randint(0, 1000, 1)[0]
    np.random.seed(seed)
    
    # draw n IID data points from N(true_mu, true_sigma)
    Xs = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

    # find estimates for sample mean, sample standard deviation
    Xbar = np.mean(Xs)
    S_squared = 1/(n-1) * np.sum((Xs-Xbar)**2)
    S = np.sqrt(S_squared)
    
    # make a true Gaussian PDF and an estimated PDF from xbar, S
    dense_xs_for_plotting = true_mu + 4 * true_sigma * np.linspace(-1, 1, 100)
    estimated_pdf = stats.norm.pdf(dense_xs_for_plotting, loc=Xbar, scale=S)
    true_pdf = stats.norm.pdf(dense_xs_for_plotting, loc=true_mu, scale=true_sigma)

    # plotting
    fig, axs = plt.subplots(2,1,figsize=(8,6), sharex=True)
    # histogram observed data
    #axs[0].hist(Xs, label=f"{n} observed $X_i$'s")
    for each in Xs:
        axs[0].axvline(each,color='k',ls='-',label='data point')
    axs[0].axvline(Xbar, ls='--', color='r', label=r'$\bar{X}$'+' = {:.2f}'.format(Xbar))
    #axs[0].set_title('histogram of observed data')
    axs[0].set_title('observed data')
    #axs[0].set_ylabel('# of data points')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.05,1), borderaxespad=0)
    # estimated vs. true PDFs
    estimated_label = "estimated:\n  N(mean={:.2f}, sd={:.2f})".format(Xbar, S)
    true_label = "true:\n  N(mean={:.2f}, sd={:.2f})".format(true_mu, true_sigma)
    axs[1].plot(dense_xs_for_plotting, estimated_pdf, label=estimated_label)
    axs[1].plot(dense_xs_for_plotting, true_pdf, color='k', label=true_label)
    axs[1].set_title('estimated vs. true PDFs')
    axs[1].set_ylabel('probability density')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05,1), borderaxespad=0)
    plt.show()
    
    # print results
    print('random seed: {}'.format(seed))
    print('inputs:')
    print('\tn: {:d}'.format(n))
    print('\ttrue mu: {:.2f}'.format(true_mu))
    print('\ttrue sigma: {:.2f}'.format(true_sigma))
    print('estimates:')
    print('\tXbar: {:.2f}'.format(Xbar))
    print('\tS: {:.2f}'.format(S))
    print(f'{n} generated data points:')
    print(f'\t{Xs}')
    
    
def get_dist_of_xbar_and_S(n, true_mu, true_sigma, n_experiments=10000):
    '''
    Given:
        a number of data points to sample, n
        a true mean of a Gaussian, true_mu
        a true standard deviation of a Gaussian, true_sigma
        a number of experiments/samples to generate
    Generates n_experiments samples of n from a Gaussian with supplied parameters
    Plots histograms of the sample mean, xbar, and 
        the sample estimate of the population standard deviation, S
    Returns three arrays: the samples, the sample means, and the sample standard deviation Ss
    '''
    all_Xs = []; all_Xbars = []; all_Ses = []
    for i in range(n_experiments):
        # generate sample
        Xs = np.random.normal(loc=true_mu, scale=true_sigma, size=n)
        # compute sample mean
        Xbar = np.mean(Xs)
        # compute sample variance
        S_squared = 1/(n-1) * np.sum((Xs-Xbar)**2)
        # standard deviation = sqrt(variance
        S = np.sqrt(S_squared)
        
        all_Xs.append(Xs); all_Xbars.append(Xbar); all_Ses.append(S)
    all_Xs = np.array(all_Xs); all_Xbars = np.array(all_Xbars); all_Ses = np.array(all_Ses)
    
    # plot histograms of xbar, S
    fig, axs = plt.subplots(1,2,figsize=(11,4))
    axs[0].hist(all_Xbars, bins=30, density=True)
    axs[0].axvline(true_mu, color='k', ls='--', label='true $\mu$')
    axs[0].legend()
    axs[0].set_xlabel(r'$\bar{x}$')
    axs[0].set_ylabel('fraction of samples')
    axs[1].hist(all_Ses, bins=30, density=True)
    axs[1].axvline(true_sigma, color='k', ls='--', label='true $\sigma$')
    axs[1].legend()
    axs[1].set_xlabel(r'$S$')
    axs[1].set_ylabel('fraction of samples')
    plt.suptitle('{} samples of {} $X$\'s $\sim N({:.2f}, {:.2f})$'.format(n_experiments, 
                                                                        n, true_mu, true_sigma))
    plt.subplots_adjust(wspace=0.4)
    plt.show()
    
    return all_Xs, all_Xbars, all_Ses
