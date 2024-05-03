from scipy.stats import binom_test

# Number of trials (sample size)
n = 30

# Number of successes (number of people you found to be "dumb")
x = 2

# Hypothesized probability of success under the null hypothesis (your friend's claim)
p_hypothesized = 0.9  # You may adjust this based on your friend's statement

# Perform the binomial test
p_value = binom_test(x, n, p_hypothesized)

# Print the p-value
print(f"P-value: {p_value}") # 8.67e-07

# which means:
# A: The probability of observing 2 or fewer dumb people in a sample of 30 people is 8.67e-07
# the conclusion is that the null hypothesis is rejected because the p-value is less than 0.05
