# QRCode.py
# Locate the pattern in images.



import math
import operator
import numpy as np
import cv2				#OpenCV version 2.4.11

import utilities as ut

# Import all global constants and coefficients
from settings import *

### Constants & Coefficients ###

thresh = 200
thresh_max_value = 255
canny_thresh1 = 100
canny_thresh2 = 200



def getVertices(contour, marker_center):
	'''
	Find the four vertices of marker in its corresponding contour.

	Input:
		contour: Marker contour - numpy.array
		marker_center: Marker center point - numpy.array
	Output:
		Array of vertices of marker - numpy.array
	'''

	# Find the min rotated rect of 'contour'
	# We divide the contour and min rect into four quadrants
	minRect = cv2.minAreaRect(contour)

	# Get vertices of the min rect 
	rectVertices = cv2.cv.BoxPoints(minRect)

	dists_to_marker_center = np.zeros((4, 1))
	vertices = np.array([None] * 4)

	for P in contour:
		P = P[0]

		dists_to_rect_vertices = []
		for rectVertex in rectVertices:
			rectVertex = np.array(rectVertex)
			dists_to_rect_vertices.append(ut.distanceP2P(P, rectVertex))

		# Determine which quadrant that P locates in
		section_idx = np.argmin(dists_to_rect_vertices)

		dist_to_marker_center = ut.distanceP2P(P, marker_center)

		if dist_to_marker_center > dists_to_marker_center[section_idx]:
			dists_to_marker_center[section_idx] = dist_to_marker_center
			vertices[section_idx] = P

	return vertices



def updateVerticesOrder(vertices, marker_center, pattern_center):
	'''
	Reorder the vertices so that the first point is the corner vertex of the pattern.

	Input:
		vertices: Vertices of the marker - numpy.array
		marker_center: Center of the marker - numpy.array
		pattern_center: Center of the pattern - numpy.array
	Output:
		New vertices in right order - numpy.array
	'''
	dists = []
	for i in range(len(vertices)):
		dists.append((i, abs(ut.distanceP2L(marker_center, pattern_center, vertices[i]))))

	dists = sorted(dists, key=operator.itemgetter(1))

	corner_idx = dists[0][0] if ut.distanceP2P(vertices[dists[0][0]], pattern_center) \
		> ut.distanceP2P(vertices[dists[1][0]], pattern_center) else dists[1][0]

	return np.append(vertices[corner_idx:], vertices[:corner_idx])



def detectQRCode(src):
	'''
	Extract four QR code corner coordinates from the input image src.

	Input:
		src: input image - np.ndarray
	Output:
		vertices: Image coordinates of QR code corners in the order of 
					Top Left -> Top Right -> Bottom Right -> Bottom Left 
					(w.r.t the code itself not the image)
				- np.array
	'''

	# Grayscale image
	gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	# Threshold
	th, thresh_img = cv2.threshold(gray_img, thresh, thresh_max_value, cv2.THRESH_BINARY)

	# Canny to extract edges
	canny_img = cv2.Canny(thresh_img, canny_thresh1, canny_thresh2)

	# Find contours
	contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hierarchy = hierarchy[0]

	'''
	Find indices of the three markers of QR code in list 'contours'.
	Store the indices in list 'marker_candidate'.
	Only three elements should be included.
	'''
	marker_candidate = []
	for i in range(len(hierarchy)):
		j, count = i, 0
		# 'hierarchy[i]' format: [Next, Previous, First_Child, Parent]
		while hierarchy[j][2] != -1:
			# Keep searching for child in next level
			j = hierarchy[j][2]
			count += 1
		# Markers have nested contours with a total of six levels
		if count == 5:
			marker_candidate.append(i)

	if len(marker_candidate) < 3:
		raise MarkerNumError('Number of detected markers is less than 3')
	else:
		marker_candidate = marker_candidate[-3:]


	'''
	Compute moments of contours to obtain the mass centers.
	'''
	mass_centers = []
	for contour in contours:
		M = cv2.moments(contour)
		if M['m00'] == 0:
			mass_centers.append((0, 0))
		else:
			mass_centers.append(np.array((int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))))


	'''
	Find which one of the three candidates is Top Left marker, Top Right marker and Bot Left marker, respectively
	'''
	A = marker_candidate[0]
	B = marker_candidate[1]
	C = marker_candidate[2]

	AB = ut.distanceP2P(mass_centers[A], mass_centers[B])
	BC = ut.distanceP2P(mass_centers[B], mass_centers[C])
	AC = ut.distanceP2P(mass_centers[A], mass_centers[C])

	top_left, top_right, bot_left = None, None, None
	P, Q = None, None

	# In triangle ABC, the vertex not involved in the longest side is the mass center of 'top_left' contour.
	if AB > BC and AB > AC:
		top_left, P, Q = C, A, B
	elif BC > AB and BC > AC:
		top_left, P, Q = A, B, C
	elif AC > AB and AC > BC:
		top_left, P, Q = B, A, C

	d = ut.distanceP2L(mass_centers[P], mass_centers[Q], mass_centers[top_left])
	slope = ut.slope(mass_centers[P], mass_centers[Q])

	if slope is None:
		bot_left, top_right = P, Q
	elif slope < 0 and d < 0:
		bot_left, top_right = P, Q
	elif slope > 0 and d < 0:
		top_right, bot_left = P, Q
	elif slope < 0 and d > 0:
		top_right, bot_left = P, Q
	elif slope > 0 and d > 0:
		bot_left, top_right = P, Q


	pattern_center = None
	top_left_vertices, top_right_vertices, bot_left_vertices = None, None, None

	# Get the pattern center
	pattern_center = np.array(((mass_centers[P][0] + mass_centers[Q][0]) // 2, \
		(mass_centers[P][1] + mass_centers[Q][1]) // 2))

	# Get markers vertices
	top_left_vertices = getVertices(contours[top_left], mass_centers[top_left])
	top_right_vertices = getVertices(contours[top_right], mass_centers[top_right])
	bot_left_vertices = getVertices(contours[bot_left], mass_centers[bot_left])

	top_left_vertices = updateVerticesOrder(top_left_vertices, mass_centers[top_left], pattern_center)
	top_right_vertices = updateVerticesOrder(top_right_vertices, mass_centers[top_right], pattern_center)
	bot_left_vertices = updateVerticesOrder(bot_left_vertices, mass_centers[bot_left], pattern_center)


	'''
	Find the fourth Bottom Right corner of the pattern.
	'''
	M1, M2 = None, None
	bot_right_corner = None

	M1 = top_right_vertices[1] if ut.distanceP2P(top_right_vertices[1], mass_centers[top_left]) \
		> ut.distanceP2P(top_right_vertices[-1], mass_centers[top_left]) else top_right_vertices[-1]
	M2 = bot_left_vertices[1] if ut.distanceP2P(bot_left_vertices[1], mass_centers[top_left]) \
		> ut.distanceP2P(bot_left_vertices[-1], mass_centers[top_left]) else bot_left_vertices[-1]

	bot_right_corner = ut.intersection(top_right_vertices[0], M1, bot_left_vertices[0], M2)

	if bot_right_corner is None:
		raise IntersectionError('Bottom line and right line do not intersect')

	return np.array([top_left_vertices[0], top_right_vertices[0], bot_right_corner, bot_left_vertices[0]])


