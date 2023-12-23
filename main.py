from src.cs_qst import CSQST
from src.bw_qst import BWQST
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import product
from functools import reduce
import matplotlib.cm as cm
import scipy

L = 3
psi = np.zeros(2**L)
psi[0] = 1/np.sqrt(2)
psi[-1] = 1/np.sqrt(2)

cs = CSQST(L, psi)
bw = BWQST(L, psi)
# cs.run(100, 15, epsilon=0.4)

# bw.run_avg(3, 100, 15)
# bw.show_output("BW")

m_min = 1
m_max = 20

n_msettings = np.arange(m_min, m_max, 1)
f_avgs = np.zeros(len(n_msettings))
f_stds = np.zeros(len(n_msettings))

i = 0
for m in n_msettings:
    # f_avgs[i], f_stds[i] = cs.run_avg(3, 100, m, epsilon=0.45)
    f_avgs[i], f_stds[i] = cs.run_avg(4, 100, m, epsilon=0.6)
    i += 1


fig, ax = plt.subplots(constrained_layout=True)
ax.plot(n_msettings, f_avgs)
ax.scatter(n_msettings, f_avgs)
# ax.errorbar(n_msettings, f_avgs, yerr=f_stds)
ax.set_ylim(0, 1)
ax.set_xlabel("m")
ax.set_ylabel("F")
ax.set_title(r"CS $L=3$ GHZ")
plt.show()
# plt.savefig("test3.png")
np.savetxt("cs_msweep2.csv", f_avgs, delimiter=',')