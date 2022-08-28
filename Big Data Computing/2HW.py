#!/usr/bin/env python
# coding: utf-8

# In[96]:


#!/usr/bin/env python
# coding: utf-8


import time
import sys
import math
import numpy as np

# function we use to read from files and create the pointset P
def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result

# function we use to compute the pair-wise distances between the points of P
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)


def SeqWeightedOutliers(P,W,k,z,alpha):
    
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
    
    r_min = dists/2.0         
    r = r_min  
    num_guesses = 1
    
      
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

            Max = 0 # M will store the maximum ball weight            
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
            S = [P[i] for i in S_indexes] # set of centers found by the algorithm            
            return (S_indexes, S, Z, r_min, r, num_guesses, time.time()-start)
        else:
            r = 2*r
            num_guesses += 1        
      
        
def ComputeObjective(P,S,z):
  
    centers_indexes = S[0].copy()    
    distances=np.zeros((len(P), len(centers_indexes)))
    dist=[]
    # we follow the hint: for each point P[i] in P we compute the distances from all the centers,
    # and then save in dist[i] the minimum of such distances, i.e. euclidean(P[i], {set of centers}).
    # At the end we sort dist, discard the z-largest values and then return the last element of the list.
    for i in range(len(P)):        
        for j in range(len(centers_indexes)):            
            distances[i, j] = euclidean(P[i],P[centers_indexes[j]])          
        dist.append(np.min(distances[i, :]))
    dist.sort()    
    return dist[len(P)-z-1]




def main():
    
    # read input 
    file_path = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    
    inputPoints = readVectorsSeq(file_path)
    weights = [1 for j in range(len(inputPoints))]

    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    
    objective = ComputeObjective(inputPoints,solution,z)

    print(f'\nInput size n = {len(inputPoints)}')
    print(f'Number of centers k = {k}')
    print(f'Number of outliers z = {z}')
    print(f'Initial guess = {solution[3]}')
    print(f'Final guess = {solution[4]}')
    print(f'Number of guesses = {solution[5]}')
    print(f'Objective function = {objective}')
    print(f'Time of SeqWeightedOutliers = {solution[6]*1000}')
    
if __name__ == "__main__":
    main()
        

# In[ ]:

