import numpy as np
import matplotlib.pyplot as plt

def np_str_into_function(np_str):
    """Convert a numpy function string into a function."""
    new_str = np_str.replace("np.", "")
    return f"f(x) = {new_str}"

# fig = plt.figure()
# ax = plt.axes(projection='3d')
x = np.linspace(0, 1, 500)
y = np.linspace(0,1, 500)
x, y = np.meshgrid(x, y)
z_str = 'np.cos(4*x + y) + np.cos(-3*y)'
z = eval(z_str)

z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()
fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, z, cmap='gray', vmin=z_min, vmax=z_max)
ax.set_title(np_str_into_function(z_str))
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.show()
# ax.contour3D(X, Y, Z, 100, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(90, 0)
# plt.show()