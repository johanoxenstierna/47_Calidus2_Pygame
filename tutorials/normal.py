import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
from scipy import stats

adf = np.random.normal()

# beta_pdf = beta.pdf(x=np.linspace(0, 1, 100), a=2, b=5, loc=0)
# ax0 = plt.plot(beta_pdf)

# rvs = np.random.normal(loc=640, scale=150, size=1000)
# plt.hist(rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

# """WEIGHTED LINSPACE"""
# # your distribution:
# distribution = stats.norm(loc=50, scale=5)
#
# # percentile point, the range for the inverse cumulative distribution function:
# bounds_for_range = distribution.cdf([0, 100])
#
# # Linspace for the inverse cdf:
# pp = np.linspace(*bounds_for_range, num=100)
#
# x = distribution.pdf(pp)
# # x = distribution.cdf(pp)
# # x = distribution.ppf(pp).astype(float)
#
# x[-1] = x[-2]
# # And just to check that it makes sense you can try:
# from matplotlib import pyplot as plt
# plt.plot(x)
# # plt.hist(x)
# plt.show()

# ================


# Define the normal distribution
distribution = stats.norm(loc=25, scale=10)
x = np.linspace(0, 50, 100)
y = distribution.pdf(x)
plt.plot(x, y, 'k', linewidth=2)
plt.title('Histogram of Clipped Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()


