import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm
import scipy
from src.helpers_distributions import min_max_normalize_array


NUM = 120
pdf = beta.pdf(x=np.arange(0, NUM), a=3, b=8, loc=0, scale=NUM)
pdf = min_max_normalize_array(pdf, y_range=[125, 255])

ax0 = plt.plot(pdf, marker='o')
# ax0 = plt.plot(shift_post_peak_pdf, marker='o')

# beta_rvs = beta.rvs(a=2, b=5, loc=-2, scale=4, size=1500)
# plt.hist(beta_rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

plt.show()





