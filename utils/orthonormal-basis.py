import numpy as np
import matplotlib.pyplot as plt

def point_in_other_basis(point, basis):
    """Transform a point from one basis to another."""
    return np.dot(basis, point)

new_point_e1 = point_in_other_basis(np.array([1, 0]), 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]))
new_point_e2 = point_in_other_basis(np.array([0, 1]), 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]))

t = np.linspace(0, 2*np.pi, 100)
x = np.cos(3*t)
y = np.sin(2*t)
fig, axs = plt.subplots(2,1, sharey=True, sharex=True);

axs[0].plot(x, y)
# axs[0].set_aspect();
axs[0].set_title("Basis {[1, 0],[0, 1]}")
c = np.dstack((x, y))[0]


z = np.dot(c, 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]))
axs[1].set_title("Basis {[[1, 1], [1, -1]]/sqrt(2)}")
axs[1].plot(z[:,0], z[:,1])

fig.suptitle("Orthogonal basis transformation of x=cos(3t), y=sin(2t), t in [0, 2pi]")
fig.subplots_adjust(top=0.83)
plt.show()