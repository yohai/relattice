import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib
import relattice as rl


RS = np.random.RandomState(34883482)
indx, indy = np.meshgrid(np.arange(6)-2, np.arange(6)-2)
x = indx + indy/2
y = indy.copy()
x = x + RS.randn(*x.shape)/10
y = y + RS.randn(*y.shape)/10
x = x.ravel()
y = y.ravel()
points = np.array([x, y]).T

dtri = Delaunay(points)
t = dtri.points[dtri.simplices]
d = t - np.roll(t, 1, axis=1)
max_edge_len = np.linalg.norm(d, axis=2).max(axis=1)
bad_triangles = np.where(max_edge_len > 1.5)[0]
tri = matplotlib.tri.Triangulation(x, y, triangles=dtri.simplices)

ij = rl.relattice(dtri, 35, bad_triangles=bad_triangles)

triangle_centers = dtri.points[dtri.simplices].mean(1)
# fig, ax = plt.subplots(figsize=(8,10))
# ax.triplot(tri)
# ax.plot(points[:,0], points[:,1], 'o')
# for i, p in enumerate(dtri.points):
#     ax.text(*p, f'{i}',fontsize=10, ha='left')
# for i, p in enumerate(triangle_centers):
#     ax.text(*p, f'{i}',fontsize=10,  c='r', ha='center', va='center')
# ax.set_aspect(1)

fig, ax = plt.subplots(1,1, figsize=(9,9))
ax.triplot(tri)
for lp, ep in zip(dtri.points, ij):
    if not all(np.isnan(ep)):
        ax.text(*lp, tuple(ep.astype('i').tolist()))
for i, p in enumerate(dtri.points):
    ax.text(*p, f'{i}',fontsize=13, ha='right', c='r')
for i, p in enumerate(triangle_centers):
    ax.text(*p, f'{i}',fontsize=10,  c='g', ha='center', va='center')
ax.set_aspect(1)