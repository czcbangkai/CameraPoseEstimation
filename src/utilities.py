# utilities.py
# Some utility functions for other files to use.



import math
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import cv2				#OpenCV version 2.4.11

# Import all global constants and coefficients
from settings import *



class Arrow3D(FancyArrowPatch):
	'''
	3D arrow class can be shown in matplotlib 3D model.
	'''
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)



def getImageName(image_path):
	'''
	Obtain image name given file path.
	
	Input:
		image_path: image file path - string
	Output:
		Image name with suffix - string
	'''
	dirs = image_path.split('/')
	return dirs[-1] if len(dirs) > 0 else ''



def distanceP2P(P, Q):
	'''
	Get Euclidean distance between two points.
	
	Input:
		P: Point1 - numpy.array
		Q: Point2 - numpy.array
	Output:
		Euclidean distance between P and Q - float, always positive
	'''
	P = P.astype(np.float32, copy=False)
	Q = Q.astype(np.float32, copy=False)
	
	return np.linalg.norm(P - Q)



def distanceP2L(M, N, P):
	'''
	Get perpendicular distance from point P to line MN.

	Input:
		M: First point that determines the line MN - numpy.array
		N: Second point that determines the line MN - numpy.array
		P: Point whose distance to line MN to be computed - numpy.array
	Output:
		Perpendicular distance from point P to line MN - float, could be any real number
	'''
	M = M.astype(np.float32, copy=False)
	N = N.astype(np.float32, copy=False)
	P = P.astype(np.float32, copy=False)
	
	# General form for the eqation of a line: Ax + By + C = 0
	A, B, C = 0.0, 0.0, 0.0

	if N[0] - M[0] == 0:
		# Case 0: the line is vertical, x = -C
		A = 1.0
		C = -M[0]
	else:
		# Otherwise: Ax + y + C = 0
		A = -(N[1] - M[1]) / (N[0] - M[0])		
		B = 1.0									# Fix B to 1.0
		C = -(A * M[0] + M[1])

	# Perpendicular distance: (Ax' + By' + C) / sqrt(A ^ 2 + B ^ 2)
	# If distance is positive, P is above / at the right side of line MN.
	# If distance is zero, P is on line MN.
	# If distance is negative, P is below / at the left side of line MN.
	return (A * P[0] + B * P[1] + C) / math.sqrt(A ** 2 + B ** 2)



def slope(M, N):
	'''
	Get slope of line MN.

	Input:
		M: First point that determines the line MN - numpy.array
		N: Second point that determines the line MN - numpy.array
	Output:
		Case 0: Line MN is vertical
			return None
		Otherwise:
			return slope - float
	'''
	M = M.astype(np.float32, copy=False)
	N = N.astype(np.float32, copy=False)

	if N[0] - M[0] == 0:
		# Case 0: Line MN is vertical, slope = infinity
		return None
	else:
		# Otherwise: slope = (y2 - y1) / (x2 - x1)
		return (N[1] - M[1]) / (N[0] - M[0])



def cross(v1, v2):
	'''
	Get cross product of two vectors v1 and v2.

	Input:
		v1: numpy.array
		v2: numpy.array
	Output:
		Cross product of v1 and v2 - float
	'''
	v1 = v1.astype(np.float32, copy=False)
	v2 = v2.astype(np.float32, copy=False)

	# Cross product = x1 * y2 - x2 * y1
	return v1[0] * v2[1] - v1[1] * v2[0]



def intersection(M, N, P, Q):
	'''
	Get intersection of two lines (NOT line segments) MN and PQ.

	Input:
		M: First point that determines the line MN - numpy.array
		N: Second point that determines the line MN - numpy.array
		P: First point that determines the line PQ - numpy.array
		Q: Second point that determines the line PQ - numpy.array
	Output:
		Case 0: Two lines overlap
			return None
		Case 1: Two lines are parallel
			return None
		Otherwise:
			return intersection point - numpy.array
	'''
	M = M.astype(np.float32, copy=False)
	N = N.astype(np.float32, copy=False)
	P = P.astype(np.float32, copy=False)
	Q = Q.astype(np.float32, copy=False)

	v1 = N - M
	v2 = Q - P
	crossProd = cross(v1, v2)

	if (crossProd == 0):
		# Case 0 / Case 1: Two lines overlap or are parallel
		return None
	else:
		# Otherwise: find intersection point
		t1 = cross(P - M, v2) / crossProd		
		return np.array(np.int0(M + t1 * v1))


