import random
import numpy as np
import math

len_of_words,d_k,d_v=4,8,8
query=np.random.randn(len_of_words,d_k)
key=np.random.randn(len_of_words,d_k)
values=np.random.randn(len_of_words,d_v)

print("Query\n",query)
print("key\n",key)
print("value\n",values)

self_attention=np.matmul(query,key.T)     #key.T is the transpose of the key
print(query.var(),key.var(),np.matmul(query,key.T).var())    #since there is lot more difference between the  varriance 0f key , value and np.matmul(quey,key.T) we divide np.matmul(query,key.T) with the d_k to stabilize the values
scaled=np.matmul(query,key.T)/math.sqrt(d_k)
print(query.var(),key.var(),scaled.var())                     #now the difference between the variance of query,key, scaled are similar
print(scaled)


#now performin masking
#masking is required so that at thetime of decoding the currecnt words should not look at the context of other words 
#it is not required at the time of encoding since we give the sequence of inputs but at encoding it should be generating the next word so it shouldnt look at next word

#creating a triangular matrix
l=len_of_words
mask=np.tril(np.ones((len_of_words,len_of_words)))              #tril = lower triaangular matrix
print(mask)
mask[mask == 0]=-np.infty
mask[mask == 1]=0

masked_scaled=mask+scaled
print(masked_scaled)

def softmax(x):
    return(np.exp(x).T/np.sum(np.exp(x),axis=-1)).T

attention=softmax(masked_scaled)
print(softmax(scaled))          #check the difference between masked scaled and unmasked scaled,you will get to know why scaled is important
print(attention)                
new_values=np.matmul(attention,values)
print(new_values)


#putting all the statements into a single function so that it can be used in decoding too
def scaled_dot_product_attention(query,key,values,mask=None):
    d_k=query.shape[-1]
    scaled=np.matmul(query,key.T)/math.sqrt(d_k)
    if mask is not None:
        scaled=scaled +mask
    attention=softmax(scaled)
    new_values=np.matmul(attention,values)
    return attention,new_values

values,attention=scaled_dot_product_attention(query,key,values,mask=None)
print("Querys\n",query)
print("Key\n",key)
print("values\n",values)
print("new_values\n",new_values)
print("attention\n",attention)