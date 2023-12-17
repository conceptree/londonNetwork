import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# First plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

# Second plot
plt.figure()
plt.plot([4, 5, 6], [1, 2, 3])
plt.show()
