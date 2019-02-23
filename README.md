# testwarp
test script for troubleshooting

Some things to note: i am using Python3 & opencv-python3.4

description for whats happening in the code.

def similarityTransform:
	This function matches the two corners of the eyes in both Smile Neutral & Smile Average images so that i can grab the offset when i use "gettingpointsOffset" function later on.

gettingpointsOffset is run 2 times to get the arrays of 68 landmarks for the neutral expression and the smiling one but now with their eyes aligned.

Taking the Smilingarray of points - neutral array of points we get the offset, how many pixels etc the cheek, mouth move by x and y coordinates.

The results come out as this:

	[[  1.  -1.]
	 [  0.  -1.]
	 [  0.  -1.]
	 [ -1.  -1.]
	 [ -1.   0.]
	 [  0.   0.]
	 [ -1.   3.]
	 [  0.   3.]
	 [  1.   5.]
	 [  3.   4.]
	 [  3.   2.]
	 [  3.   0.]
	 [  4.  -2.]
	 [  3.  -3.]
	 [  3.  -3.]
	 [  1.  -3.]
	 [  1.  -3.]
	 [ -1.   2.]
	 [ -2.   2.]
	 [ -2.   2.]
	 [ -1.   0.]
	 [  0.   1.]
	 [  1.   1.]
	 [  1.   0.]
	 [  1.   1.]
	 [  1.   1.]
	 [  1.   2.]
	 [ -1.   0.]
	 [ -1.   1.]
	 [ -1.   1.]
	 [ -1.   2.]
	 [ -3.  -2.]
	 [ -2.  -1.]
	 [ -1.   0.]
	 [  0.  -1.]
	 [  2.  -1.]
	 [  0.   0.]
	 [  0.   1.]
	 [ -1.   1.]
	 [ -1.   0.]
	 [ -1.  -1.]
	 [  0.  -2.]
	 [  0.   0.]
	 [  0.   1.]
	 [ -1.   1.]
	 [  0.   0.]
	 [  0.  -1.]
	 [  1.  -1.]
	 [-12. -10.]
	 [ -7.  -6.]
	 [ -4.  -2.]
	 [ -1.  -3.]
	 [  2.  -2.]
	 [  6.  -5.]
	 [ 10. -10.]
	 [  7.  -3.]
	 [  2.   1.]
	 [ -1.   1.]
	 [ -4.   1.]
	 [ -8.  -2.]
	 [-11.  -8.]
	 [ -4.  -5.]
	 [  0.  -6.]
	 [  2.  -5.]
	 [ 10.  -9.]
	 [  3.   3.]
	 [  0.   3.]
	 [ -4.   2.]]

Which makes sense since the there should only be slight shifts to start showing a smile. Now that we have these 68 off set points, theoretically when we add this onto any existing 68 facial points it should either change a neutral face into a smile or a smiling face perhaps into an even bigger smile.
And the theory checks out.

Now that the offset points are acquired, i read the landmarks for 1a.jpg, i create a Subdiv2D for this 1a, and i create a second set of points for the 'smiling 1a' by adding the offset points with the 1a points thus creating 'points' and 'pointsforwarp'. Ive checked the dimensions for both and they seem to be fine.

I even drew the delaunay triangles for both the trianglelist of the 1a and the trianglelist for 1a smiling which is a screenshot in the repo.

now that i have both trianglelists, they draw perfectly as delaunay triangles, the code for warping the actual image fails.

Getting either one of these ValueErrors depending on the image.

could not broadcast input array from shape()
operands could not be broadcast together with shapes ()







