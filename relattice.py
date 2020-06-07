
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib

def get_opposite_point(dtri, bad_triangles, triangle_index, point_index):
    """Calculates the point opposite to a given point in a delaunay triangulation.
    
    Args:
        dtri: scipy.spatial.Delaunany triangulation.
        bad_triangles: list of ints denoting triangles in dtri.simplices that should
            not be considered.
        ij: np.array of shape [n_points, 2]. The k-th row is the ij coordinate of the 
            k-th point.
        triangle_index: int. Index of triangle in dtri.simplices along which opposite 
            is calculated.
        point_index: int. Index of point in dtri.points to calculate opposite of.
    """
    triangle = dtri.simplices[triangle_index]
    if point_index not in triangle:
        # This shouldn't happen
        raise ValueError("Point not in triangle")
    
    opp_triangle_index = dtri.neighbors[triangle_index][triangle==point_index]
    if opp_triangle_index == -1:
        return "Reached boundary", None
    if bad_triangles is not None and opp_triangle_index in bad_triangles:
        return "Reached bad triangle", None
    opp_triangle = dtri.simplices[opp_triangle_index]
    opp_point_index = opp_triangle[dtri.neighbors[opp_triangle_index]==triangle_index]
    
    return opp_point_index[0], opp_triangle_index[0]

def relattice(dtri, first_triangle_index, bad_triangles=None, verbose=0, max_steps=None):
    ij = np.full((len(dtri.points), 2), np.nan)
    ij[dtri.simplices[first_triangle_index]] = [[0, 0], [0, 1], [1, 0]]
    
    # A "step" is a tuple of the form (triangle_index, point_index) where the indices 
    # refer to the position of the triangle in dtri.simplices and the point in dtri.point.
    steps_to_take = {(first_triangle_index, pi) 
                     for pi in dtri.simplices[first_triangle_index]}
    steps_done = set()
    
    n_steps=0
    if max_steps is None:
        max_steps = len(dtri.points) * 3
    while steps_to_take and n_steps < max_steps:
        n_steps += 1
        step = steps_to_take.pop()
        steps_done.add(step)
        triangle_index, point_index = step
        if verbose >=1:
            print(f'taking step={step}', end=' : ')
        
        triangle = dtri.simplices[triangle_index]
        opp_point_index, opp_triangle_index = get_opposite_point(dtri, 
                                                                 bad_triangles,
                                                                 triangle_index,
                                                                 point_index)
        
        # triangle, but rolled to put point_index in the first entry
        rolled_triangle = np.roll(triangle, -np.argmax(triangle  ==  point_index)) 
        opp_ij = [-1,1,1] @ ij[rolled_triangle]  # this is equivalent to row2 + row1 - row0
        if opp_triangle_index is None:  # reached boundary/bad triangle
            if verbose >= 1:
                print(opp_point_index)
            continue
        if all(ij[opp_point_index] == opp_ij):
            if verbose >= 2 :
                print(f'closed a loop at {opp_ij}')
            continue
        if not all(np.isnan(ij[opp_point_index])):
            print(f"COLLISION AT point #{opp_point_index}. "
                  f"Triangle #{triangle_index} gives {opp_ij}. "
                  f"Already assigned {ij[opp_point_index].astype('i').tolist()}.")
            return ij

        if verbose >= 1:
            print(f'Solved! point #{opp_point_index} is {opp_ij.astype("i").tolist()}')
        ij[opp_point_index] = opp_ij
        
        # add also opposite step to done, as it's redundant.
        steps_done.add((opp_triangle_index, opp_point_index))
        proposed_steps = set((opp_triangle_index,  point_index) 
                             for point_index in dtri.simplices[opp_triangle_index])
        proposed_steps = proposed_steps - steps_done
        steps_to_take.update(proposed_steps)
        if verbose >=2 and proposed_steps:
                print(f'adding step {proposed_steps}', end=',')
        if verbose >= 1:
            print()

    if not np.isnan(ij).any():
        ij = ij.astype('i')
    return ij
           