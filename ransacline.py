import numpy as np
import matplotlib.pyplot as plt
import random #I'm pretty sure random is a standard python library https://docs.python.org/2/library/
import math

#Generates noisy data points for a line
#Using form y=mx+c
#m=slope
#c=y-intercept
#noise=std deviation for noise threshold
def GenerateLine(m,c,noise,num_off_points):
    distribution=np.random.normal(0,noise,num_off_points)
    x_points=np.linspace(10,200,num_off_points)
    xs=[]
    ys=[]
    for i in range(0,num_off_points):
        xs.append(x_points[i])
        y=m*x_points[i]+c
        y=y+distribution[i]
        ys.append(y)
    return xs,ys

#Generates a line without any noise
def GenerateLineNoNoise(m,c,num_off_points):
    x_points=np.linspace(10,200,num_off_points)
    xs=[]
    ys=[]
    for i in range(0,num_off_points):
        xs.append(x_points[i])
        y=m*x_points[i]+c
        ys.append(y)
    return xs,ys


#Generates the noisy dataset with inliers and outliers
def CreateProblem():
    noise=20
    slope=7
    y_intercept=4
    num_of_points=300
    x,y=GenerateLine(slope,y_intercept,noise,500)

    outlier_x,outlier_y=GenerateLine(12,100,30,20)
    x=x+outlier_x
    y=y+outlier_y
    outlier_x2,outlier_y2=GenerateLine(-3,-80,30,20)
    x=x+outlier_x2
    y=y+outlier_y2
    return x,y

#DO NOT MODIFY ANY CODE ABOVE THISE LINE

#Returns the dataset containing N random points
def chooseRandom(x, y, N):
    x_sample, y_sample = zip(*random.sample(list(zip(x, y)), N)) #googled this
    return x_sample, y_sample

#return the m and c values for a set of data points
#least squares approach:
def linearFit(x, y):
    N = len(x)
    x = np.array(x) #convert to arrays for easier math
    y = np.array(y)
    xx = x * x
    xy = x * y
    c = ((sum(y) * sum(xx)) - (sum(x) * sum(xy))) / ((N * sum(xx)) - (sum(x)**2))
    m = ((sum(xy)) - (N * np.mean(x) * np.mean(y))) / ((sum(xx) - N*(np.mean(x)**2)))

    #print("m = " + str(m))
    #print("c = " + str(c) + "\n")
    return m, c

#Minimize the sum of distances from line
#Sum over all (x_i, y_i)
def findError(x, y, m, c):
    sum = 0
    for i in range(0, len(x)):
        num = abs(m*x[i] - y[i] + c)
        den = math.sqrt(m**2 + 1)
        sum += (num / den)

    return sum

#x = x coordinates of the datapoints
#y = y coordinates of the datapoints
def RANSAC(x,y):
    solution_m=0
    solution_c=0

    ##Your RANSAC solution here
	##You may define additional functions as necessary
	## Libraries are limited to the already added ones and the standard python ones

    # Randomly select N points. In CreateProblem(), the number of points seems to be 300. I'll start with 75, and 1000 iterations
    N = 75;
    iterations = 1000
    min_error = float("inf") #set the initial error to be effectively infinite

    for i in range(1, iterations):
        x_sample, y_sample = chooseRandom(x, y, N)
        current_m, current_c = linearFit(x_sample, y_sample)
        error = findError(x, y, solution_m, solution_c)
        if (error < min_error):
            solution_m, solution_c = current_m, current_c
            min_error = error
            #print("minimum error = " + str(min_error))

    print("m = " + str(solution_m))
    print("c = " + str(solution_c))
    print("final error = " + str(min_error))
    return solution_m,solution_c

def main():
    x,y=CreateProblem()

    sol_m,solution_c=RANSAC(x,y)

    sol_x,sol_y=GenerateLineNoNoise(sol_m,solution_c,30)

    #Plot data and solution
    plt.plot(x,y,'ro')
    plt.plot(sol_x,sol_y,linewidth=2.0)
    plt.show()

main()
