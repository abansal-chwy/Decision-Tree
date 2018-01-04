
# coding: utf-8

# In[404]:

import numpy as np
import matplotlib.pyplot as plt

test = np.loadtxt("train.txt", delimiter=',')


def entropy(D):                                                                               #entropy of the dataset
    class1=0.0
    class2=0.0
    for i in D[:,-1]:
        if(i==0):
            class1=class1+1
        else:
            class2=class2+1
    total=class1+class2
    entropy = -((class1/total)*np.log2(class1/total)) - ((class2/total)*np.log2(class2/total))
     
    return entropy





# # QUESTION 1

# In[405]:

def IG(D, index, value):                                                                #information gain of the Dataset
    count1=D[:,[index,-1]]
    total_entropy = entropy(test)
   
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
    
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
   # print class1_0,class1_1,class2_0,class2_1  
    class_less_value=class1_0+class1_1
    class_more_value=class2_0 +class2_1
    total = class_less_value + class_more_value
    
    if(class1_0!=0):
        t1=((class1_0)/(class1_0+class1_1))*np.log2((class1_0)/(class1_0+class1_1))
    else:
        t1=0
    if(class1_1!=0):    
        t2=((class1_1)/(class1_0+class1_1))*np.log2((class1_1)/(class1_0+class1_1))
    else:
        t2=0
    
    if(class2_0!=0): 
        t3=((class2_0)/(class2_0+class2_1))*np.log2((class2_0)/(class2_0+class2_1))
    
    else:
        t3=0
       # t4=0
    if(class2_1!=0):    
        t4=((class2_1)/(class2_0+class2_1))*np.log2((class2_1)/(class2_0+class2_1))
    
    else:
        t4=0
        
        
        
    try:entropy_0 = - (t1) - (t2)   
    except ZeroDivisionError:entropy_0 =0
        
        
    try: entropy_1 = - (t3) - (t4)       
    except ZeroDivisionError:entropy_1 =0
    
    information_gain = total_entropy - (((class_less_value/total)*entropy_0) +((class_more_value/total)*entropy_1))
    
    if(information_gain==None):
        information_gain=0
    
    return information_gain

IG(test,9,7)


# # Question 2 

# In[383]:

def G(D, index, value):
    count1=D[:,[index,-1]]
    #val_freqs = np.unique(count1[:,0], return_counts=True)
    #print val_freqs
   
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
    
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
   # print class1_0,class1_1,class2_0,class2_1  
    class_less_value=class1_0+class1_1
    class_more_value=class2_0 +class2_1
    total = class_less_value + class_more_value
    
    try:t1=class1_0/class_less_value
    except ZeroDivisionError:t1 =0
        
    try:t2=class1_1/class_less_value
    except ZeroDivisionError:t2=0    
        
    try:t3=class2_0/class_more_value
    except ZeroDivisionError:t3=0
    
    try:t4=class2_1/class_more_value
    except ZeroDivisionError:t4=0
    
    
    gini_1 = 1 - ((t1)**2 + (t2)**2)
    
        
    gini_2=  1 - ((t3)**2 + (t4)**2)
    
    gini_index = gini_1*(class_less_value/total) + gini_2*(class_more_value/total)

    return gini_index
G(test,0,2)


# # Question 3 

# In[384]:

def CART(D, index, value):    
    count1=D[:,[index,-1]]
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
   
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
     
    class_less_value=class1_0+class1_1
    class_more_value=class2_0 +class2_1
    total = class_less_value + class_more_value
    
    try:t1=class1_0/class_less_value
    except ZeroDivisionError:t1 =0   
    try:t2=class2_0/class_more_value
    except ZeroDivisionError:t2 =0
    try:t3=class1_1/class_less_value
    except ZeroDivisionError:t3 =0
    try:t4=class2_1/class_more_value
    except ZeroDivisionError:t4 =0    
    
    cart_total = 2*(class_less_value/total)*(class_more_value/total)*(abs(t1-t2)+abs(t3-t4))
  
    return cart_total
CART(test,7,4)


# #  Question 4

# In[385]:

def bestSplit(D, criterion):
    best_gini =G(test,0,0)
    list_gini=0.0
    list_IG=0.0
    best_IG=0.0
    list_CART=0.0
    best_CART=0.0
    BS=()
    
    if (criterion=="GINI"):
        for i in range(0,10):
            for j in np.unique(D[:,i]):
                list_gini=G(test,i,j)
               
                if list_gini < best_gini:
                    best_gini=list_gini
                    index = i
                    value=j
        l = list(BS)
        l.append(index)
        l.append(value)
        BS = tuple(l)
        
        return BS
        
    elif (criterion=="IG"):
        for i in range(0,10):
            for j in np.unique(D[:,i]):
                list_IG=IG(test,i,j)
                if list_IG > best_IG:
                    best_IG=list_IG
                    index = i
                    value=j
        l = list(BS)
        l.append(index)
        l.append(value)
        BS = tuple(l)
        
        return BS
    elif (criterion=="CART"):
        
        for i in range(0,10):
            for j in np.unique(D[:,i]):
                list_CART=CART(test,i,j)
               
                if list_CART > best_CART:
                    best_CART=list_CART
                    index = i
                    value=j
        l = list(BS)
        l.append(index)
        l.append(value)
        BS = tuple(l)
        
        return BS
bestSplit(test,"IG")


# In[ ]:




# # Question 5

# In[398]:

def load(filename):
    train = np.loadtxt(filename, delimiter=',')
    count=[]
    count1=train[:,:-1]
    count2=train[:,-1]
    count.append(count1)
    count.append(count2)
    train=tuple(count)
    
    return train
 
load("train.txt")


# # Question 6

# *Attached in the PDF*

# # Question 7

# In[409]:

def classifyIG(train, test):
    test = np.loadtxt(test, delimiter=',')
    train = np.loadtxt(train, delimiter=',')
    
    
    index,value=bestSplit(train,"IG")    
    #print index,value
    count1=train[:,[index,-1]]
    predicted=[]
    difference=0
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
    
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
    #print class1_0,class1_1,class2_0,class2_1  
    
    if(class1_0>=class2_0 and class1_0>=class1_1):
        majority_less_value = 0
        majority_more_value=1
    elif(class1_1>=class2_1 and class1_1>=class1_0):   
        majority_less_value = 1
        majority_more_value=0
    elif(class2_0>=class1_0 and class2_0>=class2_1):
        majority_less_value = 1
        majority_more_value=0
    elif(class2_1>=class1_1 and class2_1>=class2_0):   
        majority_less_value = 0
        majority_more_value=1
    #print majority_less_value,majority_more_value
    
    for i in range(0,len(test)):
        if(test[i,index]<=value):
            predicted.append(majority_less_value)
        else:
            predicted.append(majority_more_value)
    for i in range(0,len(test)):
        if(test[i,-1]!=predicted[i]):
            difference= difference +1
    
    return predicted,difference
classifyIG("train.txt","test.txt")


# In[410]:

def classifyG(train, test):
    test = np.loadtxt(test, delimiter=',')
    train = np.loadtxt(train, delimiter=',')
    index,value=bestSplit(train,"GINI")
    #print index,value
    #values = np.unique(train[:,index])
    count1=train[:,[index,-1]]
    predicted=[]
    difference=0
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
    
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
    #print class1_0,class1_1,class2_0,class2_1  
    
    if(class1_0>=class2_0 and class1_0>=class1_1):
        majority_less_value = 0
        majority_more_value=1
    elif(class1_1>=class2_1 and class1_1>=class1_0):   
        majority_less_value = 1
        majority_more_value=0
    elif(class2_0>=class1_0 and class2_0>=class2_1):
        majority_less_value = 1
        majority_more_value=0
    elif(class2_1>=class1_1 and class2_1>=class2_0):   
        majority_less_value = 0
        majority_more_value=1
    #print majority_less_value,majority_more_value
    
    for i in range(0,len(test)):
        if(test[i,index]<=value):
            predicted.append(majority_less_value)
        else:
            predicted.append(majority_more_value)
    for i in range(0,len(test)):
        if(test[i,-1]!=predicted[i]):
            difference= difference +1
    
    return predicted,difference
    
classifyG("train.txt","test.txt")


# In[411]:

def classifyCART(train, test):
    test = np.loadtxt(test, delimiter=',')
    train = np.loadtxt(train, delimiter=',')
    index,value=bestSplit(train,"CART")    
    #print index,value
    count1=train[:,[index,-1]]
    predicted=[]
    difference=0
    class1_0=0.0
    class1_1=0.0
    class2_0=0.0
    class2_1=0.0
    
    for i in np.unique(count1[:,0]):
        for j in range(0,len(count1)):
           
            if (i<=value):
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class1_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class1_1 +=1
            else:
                if ((count1[j,0]==i) and (count1[j,1]==0)):
                    class2_0 +=1
                elif ((count1[j,0]==i) and (count1[j,1]==1)):
                    class2_1 +=1
    #print class1_0,class1_1,class2_0,class2_1  
    
    if(class1_0>=class2_0 and class1_0>=class1_1):
        majority_less_value = 0
        majority_more_value=1
    elif(class1_1>=class2_1 and class1_1>=class1_0):   
        majority_less_value = 1
        majority_more_value=0
    elif(class2_0>=class1_0 and class2_0>=class2_1):
        majority_less_value = 1
        majority_more_value=0
    elif(class2_1>=class1_1 and class2_1>=class2_0):   
        majority_less_value = 0
        majority_more_value=1
    #print majority_less_value,majority_more_value
    
    for i in range(0,len(test)):
        if(test[i,index]<=value):
            predicted.append(majority_less_value)
        else:
            predicted.append(majority_more_value)
    for i in range(0,len(test)):
        if(test[i,-1]!=predicted[i]):
            difference= difference +1
    
    return predicted,difference
    
    
classifyCART("train.txt","test.txt")


# In[415]:

def main():
    total_entropy = entropy(test)
    
    print("Class Entropy",total_entropy)                              # Total Entropy               
    
    print("Information gain for Index and Value ",IG(test,0,1))       #Information Gain 
    
    print("GINI Inde for Index and Value",G(test,0,1))                #Gini Index
    
    print("CART for Index and Value",CART(test,7,4))                  #CART
    
    
    #Best possible splits for Info Gain ; Gini Index and CART
    
    test2=load("train.txt")                          
    print("Best Split using Info Gain",bestSplit(test,"IG"))
    print("Best Split using Gini Index",bestSplit(test,"GINI"))
    print("Best Split using CART",bestSplit(test,"CART"))
    
    
    # Predicting classes
    
    print("Predicted Classes with DIfference Using Info Gain",classifyIG("train.txt","test.txt")) 
    
    print("Predicted Classes with DIfference Using Gini Index",classifyG("train.txt","test.txt"))
    
    print("Predicted Classes with DIfference Using CART",classifyCART("train.txt","test.txt"))
    
if __name__=="__main__": 
    main()


# In[ ]:



