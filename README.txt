This is the python version of the NodeIn interview coding challenge. The goal of this test is to see if you are able to program a version of the RANSAC algorithm. It is a well used algorithm in Computer Vision world and is even something we use here at NodeIn.

To get started you need to be on a Linux system preferably Ubuntu. The code requires access to the Libraries numpy and matplotlib. Both can be installed via this command

sudo apt-get install python-numpy python-matplotlib

Running the python program will show you a plot of the data, and also plots a line for your solution.

git is also required to download and submit your solution to the problem. If you have never used git before the 4 command you will need is
git clone
git add
git commit
git push

How exactly to use them can be easily found online

Problem Description:

For this test you will be programming a version of the RANSAC algorithm. RANSAC stands for random sample consenus and is an algorithm used to fit a model over data with outliers and inliers. 

The basic gist of RANSAC is:

For i<iterations
 	Randomly select minimum amount of points needed to fit model
	Calculate Model
	Calculate Model Error against other datapoints
	If(Error<Best_error)
		Best_Model =model

A better explanation can be found at https://en.wikipedia.org/wiki/Random_sample_consensus.

Typically at the end you can find which points are considered inliers using the Best_model. A final pass using all the inliers found is then done to get the best solution possible(Typically done via Least Squares)


For this problem you are only required to find the best model of a line defined by the equation y=mx+c. So your ransac algorithm should spit out the m and c parts of your line equation. The last step improvement via least squares
is not needed.

HINTS:
- Get the basic line equation parts working first. Then worry about the RANSAC portion.
- 2 points define a line
- Once you have a calculated line model the error between the line and a point can be calculated using the distance between a line and a point. (https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)
- Recommended error metrics are to either maximise the number of inliers(points below a threshold) or to minimise the sum of all distances between a line and the points
- Pick a low number first for your threshold, and try to get the code working with that. The actual threshold is defined by the noise term, but using it directly causes some issues

Bonus points if you have extra time:
-Modify the code so that your RANSAC solution identifies which datapoints are outliers and inliers and plot these datapoints as to seperate colors.

Rules:
You can google almost anything you need to. The only thing I ask that you do not look up is a RANSAC implementation. Pseudocode like in the wikipedia article is ok.


