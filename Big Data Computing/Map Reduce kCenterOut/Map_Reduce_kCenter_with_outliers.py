# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2]) # number of centers
    z = int(sys.argv[3]) # number of outliers
    L = int(sys.argv[4]) # number of partitions to split the RDD into

    # we want to monitor the time taken to read from the file	
    start = 0 
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# strToVector reads a string from file.csv and transforms it 
# into a tuple of floats, i.e. a numerical vector

def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# compute the squared euclidean distance between point1 and point2

def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# compute the euclidean distance between point1 and point2

def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: Map-Reduce algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# In the following we implement the Map-Reduce algorithm for k-center with outliers.
# We use a coreset based approach, namely for each partition:
# - we extract a subset of k+z+1 points (the so called coreset) running Farthest First Traversal;
# - we assign to each said point y a weight w, called the representative wieght. 
#   In practice w consists of the number of points in the partition whose closest point in the coreset is exactly y;
# - we merge togheter all the obtained coresets, thus obtaining a weighted pointset;
# - finally, we run k_center_with_outliers algorithm on the weighted pointset

def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------

    start_1R = time.time() # track time required by round 1
    # extract coreset from each partition and assign representative weight
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1)) 
    # merge all the coresets togheter: elems is a weighted pointset
    elems = coreset.collect()
    end_1R = time.time() 
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    
    start_2R = time.time() # track time required by round 2
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # now we compute the final solution running SeqWeightedOutliers with alpha=2
    solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    end_2R = time.time()
    
    print('Time taken by Round 1: ', (end_1R - start_1R)*1000, ' ms')
    print('Time taken by Round 2: ', (end_2R - start_2R)*1000, ' ms')
    return solution
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def SeqWeightedOutliers(P, W, k, z, alpha): 	# (points, weights, k, z, alpha)
    
    start = time.time() # we want to track the execution time required by the method      

    # let's precompute all the pair-wise distances. 
    # To achieve better performances we'll store all these distances in a 2-d numpy array       
    
    distances = np.zeros((len(P),len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            distances[i,j] = euclidean(P[i],P[j])
    
    
    # now let's compute the initial guess for r, minimum radius
    # notice that the diagonal elements of distances are all zeros, since each diagonal element represents the distance
    # between each point and itself. Because of this, in computing r_min we have to 'discard' the diagonal elements,
    # and in order to do so we add +inf on the diagonal
    dists = np.min(distances[:k+z+1,:k+z+1]+np.diag(np.inf*np.ones(z+k+1)))
        
    r_min = dists/2.0 		# initial radius
    r = r_min	    		# we'll update iteratively r at each iteration with a newly computed radius         
    num_guesses = 1	    	# number of r updates needed to achieve an approximate solution to the problem
    
      
    while True:
      
        # We'll follow the following strategy: 
        # we use a boolean np.array Z to account for the points of P that are still in Z, at the beginning each component
        # of Z will be equal to 1 since Z=P, i.e. all the points are in Z.
        # At each iteration, instead of removing points from Z we'll just set the respective indexes to zero.
        Z = np.ones(len(P)) # boolean vector to track points in P that are yet to be removed from Z
        w = np.array(W.copy()) # vector of weights
        S_indexes = [] # this list will contain the indexes of points in P that will be chosen as centers
        W_z = np.sum(w[Z==1]) # sum of the weights of all points in Z

        while (len(S_indexes)<k) and (W_z>0):

            # to begin, for each point in P we want to compute its ball's weight
            # and then choose as new center the point whose ball has maximum weight

            Max = 0 # Max will store the maximum ball weight            
            for i in range(len(P)):
                mask = distances[:, i]<=(1+2*alpha)*r
                
                # mask is a boolean vector where each component j is set to True iff distance(P[j],P[i])<=(1+2*alpha)*r
                
                ball_weight = np.sum(w[np.logical_and(Z==1,mask)])                
                
                # notice that (Z==1 and mask) is a boolean vector with j-th componenet equal to True iff P[j] is in Z
                # and distance(P[j],P[i])<=(1+2*alpha)*r.
                
                if ball_weight > Max:
                    Max = ball_weight
                    new_center_index = i

            # once we have found the point whose ball has maximum weight, we choose this point as new center
            # so we save its index as point in P            
            S_indexes.append(new_center_index)            

            # now we have to subtract from W_z the weights of all the points that fall in the ball of center P[new_center_index]
            # and radius = (3+4*alpha)*r, and then remove said points from Z (we'll do this by setting their respective indexes to zero).
                        
            mask2 = distances[:, new_center_index]<=(3+4*alpha)*r 
            
            # mask2 is a boolean vector where each component j is set to True iff distance(P[j],new_center)<=(3+4*alpha)*r
            
            W_z -= np.sum(w[np.logical_and(Z==1,mask2)])       
            Z[np.logical_and(Z==1,mask2)] = 0
            
            # notice that (Z==1 and mask2) is a boolean vector with j-th componenet equal to True iff P[j] is in Z
            # and distance(P[j],new_center)<=(3+4*alpha)*r.

        if (W_z<=z):
            print('Initial guess = ', r_min)
            print('Final guess = ', r)
            print('Number of guesses = ', num_guesses)
            S = [P[i] for i in S_indexes] # set of centers found by the algorithm            
            #return (S_indexes, S, Z, r_min, r, num_guesses, time.time()-start)
            return S
        else:
	    # if we are yet to find a solution, double down the radius and start all over again
            r = 2*r
            num_guesses += 1   


            
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method getCloseDistance: return distance between a point p
# and a set of centers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#
def getCloseDistance(p, centers):
    closest = centers[0]
    currdist = squaredEuclidean(p, closest)
    for i in range(1, len(centers)):
        dist = squaredEuclidean(p,centers[i])
        if dist < currdist:
            currdist = dist
    return math.sqrt(currdist)
            
            

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
    topZ1 = points.map(lambda x : getCloseDistance(x, centers)).top(z+1)
    return min(topZ1)




# Just start the main program
if __name__ == "__main__":
    main()

