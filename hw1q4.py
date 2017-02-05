import random
import numpy as np
import matplotlib.pyplot as plt

def simulate_f(N):
    '''
    Generate N independent samples from f(x)=x(1-x)/Z with Z=1/6
    Args:
        N - the number of samples required
    Returns:
        nums - a 1D numpy array of samples
    '''
    generator = random.Random()
    generator.seed() # seed using current time or OS source
    
    nums = np.zeros(N)
    three_uniform = [None]*3 # this list contains 3 numbers drawn from uniform(0,1)
    
    for i in range(N):
        # each round of loop, generate 3 uniform, sort them, and put the 2nd rank into nums
        for j in range(3):
            three_uniform[j] = generator.random()
        # sort the 3 uniform numbers
        three_uniform.sort()
        # put the 2nd rank into nums
        nums[i] = three_uniform[1]
    
    return nums

def f_dist(x):
    Z = 1.0/6.0
    return x * (1-x) * 1.0 / Z

def f_integral(x):
    Z = 1.0/6.0
    return (0.5*(x**2) - (x**3)/3.0) / Z

def plot_hist(nums,dx):
    '''
    Make a histogram of the samples contained in nums
    Plot this histogram with the pdf to see asymptotic behavior for large N
    Args:
        nums - a 1D numpy array of samples with length N
        dx - bin size of the histogram
    Returns:        
        display histogram
        draw one standard deviation error bars on the histogram values
    '''
    N = nums.size # number of samples
    numbins = np.rint( (np.amax(nums) - np.amin(nums))/dx ) 
    numbins = numbins.astype(int)
    hist_counts, bin_edges = np.histogram(nums, bins=numbins, density=False)
    
    # pj_array = hist_counts * 1.0 / nums.size() # this is the estimated pj from samples
    # but here we calculate the exact pj from closed form integrate
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    left_integral = np.apply_along_axis(f_integral, 0, left_edges)
    right_integral = np.apply_along_axis(f_integral, 0, right_edges)
    pj_array = right_integral - left_integral
    
    bin_centers = 0.5 * ( left_edges + right_edges ) # array of center points of bins
    # f_values = np.apply_along_axis(f_dist, 0, bin_centers) # evaluate f at bin_centers
    
    # f_scaled_values = integrate f over bins and scale up by number of samples
    # f_scaled_values is comparable to the count inside each bin
    f_scaled_values = N * pj_array
    
    errorbars = np.sqrt(pj_array * (1-pj_array) * N)
    
    fig, ax = plt.subplots()
    # graph the bar histogram
    ax.bar(left_edges, hist_counts, width=dx, color='r', yerr=errorbars)
    # graph the scaled f 
    ax.plot(bin_centers, f_scaled_values)
    ax.set_ylabel('Count of samples in bins')
    ax.set_title('Histogram of f(x)=x(1-x)/Z')
    
    plt.show()

def main():
    # TODO
    N = 1500
    nums = simulate_f(N)
    dx = 0.05
    plot_hist(nums,dx)

if __name__ == '__main__':   # this is the boilerplate portion
    main()