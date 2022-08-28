#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random as rand
from pyspark import SparkContext
from pyspark import SparkConf

# initial configurations
conf = SparkConf().setAppName('FirstHomeWork').setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)

# we'll use this function to get the (product,customer) pairs
def filtering(docs):
    objs = docs.split(',')    
    # we return ((prodID, customerID),1) where 1 could actually be any value since we only need the keys of these pairs
    return ((objs[1],int(objs[6])),1)            

# the following functions are the same ones we already used in the word count examples
def gather_pairs_partitions(pairs):
    pairs_dict = {}
    for p in pairs:
        productID, occurrences = p[0], p[1]
        if productID not in pairs_dict.keys():
            pairs_dict[productID] = occurrences
        else:
            pairs_dict[productID] += occurrences
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def gather_pairs(pairs):
    pairs_dict = {}
    for p in pairs[1]:
        productID, occurrences = p[0], p[1]
        if productID not in pairs_dict.keys():
            pairs_dict[productID] = occurrences
        else:
            pairs_dict[productID] += occurrences
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]
    

def main():
    
    print('Insert number of partitions K (positive integer)')
    K=-5 # just to enter the while loop
    while type(K)!=int or K<=0:        
        try:
            K=int(input())
            if K<=0:
                print("Error: K must be a positive integer. Insert again.")
        except:
            print("Error: K must be a positive integer. Insert again.")

    print('\nInsert a non-negative integer H')
    H=-5 # just to enter the while loop
    while type(H)!=int or H<0:
        try:
            H=int(input())
            if H<0:
                print("Error: H must be a non-negative integer. Insert again.")
        except:
            print("Error: H must be a non-negative integer. Insert again.")

    print('\nInsert a Country S')
    S=input()    

    print('\nInsert DataSet path file')
    data_path=input()    

    # we create the RDD reading from the file and we partition it into K partitions
    rawData=sc.textFile(data_path).repartition(K).cache()

    
    # STEP 1
    print(f'\nnumber of rows in the RDD: {rawData.count()}')

    
    # STEP 2
    # to begin we filter out the rows having Quantity<=0, since we're only interested in rows having Quantity>0 
    rawData = rawData.filter(lambda x: (int(x.split(',')[3])>0))
    
    if S=='all': # if no Country is specified
        productCustomer = (rawData.map(lambda x: filtering(x))).groupByKey().map(lambda x: x[0])
    else:
        # otherwise we additionally filter out rows having Country!=S
        rawData_filtered = rawData.filter(lambda x: (x.split(',')[7]==S))        
        productCustomer = (rawData_filtered.map(lambda x: filtering(x))).groupByKey().map(lambda x: x[0])        
    print(f'\nProduct-Customer Pairs: {productCustomer.count()}')
       
    
    # STEP 3
    productPopularity1 = (productCustomer.map(lambda x: (x[0],1))).mapPartitions(gather_pairs_partitions).groupByKey().mapValues(lambda vals: sum(vals))                

    
    # STEP 4
    productPopularity2 = (productCustomer.map(lambda x: (x[0],1))).groupBy(lambda x: (rand.randint(0,K-1))).flatMap(gather_pairs).reduceByKey(lambda x, y: x + y)     

    
    # STEP 5
    if H>0:    
        #topH = productPopularity1.sortBy(lambda x: x[1], ascending=False).collect()[:H]
        topH = productPopularity1.takeOrdered(H, key=lambda x: -x[1])
        print(f'\nTop {H} Products and their Popularities:')
        for elem in topH:
            print(f'Product: {elem[0]} Popularity: {elem[1]};', end = ' ')


    # STEP 6
    elif H==0:
        lst1 = productPopularity1.sortBy(lambda x: x[0], ascending=True).collect()
        lst2 = productPopularity2.sortBy(lambda x: x[0], ascending=True).collect()
        print('\nproductPopularity1 sorted:')
        for elem in lst1:
            print(f'Product: {elem[0]} Popularity: {elem[1]};', end = ' ')
        print('\nproductPopularity2 sorted:')
        for elem in lst2:
            print(f'Product: {elem[0]} Popularity: {elem[1]};', end = ' ')


    

if __name__ == "__main__":
    main()

