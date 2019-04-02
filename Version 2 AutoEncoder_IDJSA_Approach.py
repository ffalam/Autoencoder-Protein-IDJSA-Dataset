#!/usr/bin/env python
# coding: utf-8

# In[178]:


import pca
import numpy as np
import xml.etree.ElementTree as ET
import Bio.SVDSuperimposer
from Bio.SVDSuperimposer import SVDSuperimposer
from math import sqrt
from numpy import array, dot
import random
import operator
import os
import sys
import pickle
from sklearn.decomposition import PCA
import math
import scipy.io
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from numpy.random import seed
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import array as arr


# In[26]:


coordinate_file = 'onedtja.txt'
#print(coordinate_file)
# align the models with the first one in the file. No need to call this function if already pickled to disk.

aligned_dict, ref = pca.align(coordinate_file)
#print(aligned_dict)
print(len(aligned_dict))

with open("myfile.txt", 'w') as f:
    for key, value in aligned_dict.items():
        f.write('%s:%s\n' % (key, value))


# In[ ]:





# In[27]:


#Read the Energy List

energylist = []
	#read the energy file
with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)
#energylist = energylist[:-1]  # only for 1aly            
energyarray = np.array(energylist).reshape(-1,1)
#Xenergyarray = energyarray[0:400,:]
print(energyarray.shape)


# In[28]:


#converting Aligned dictionary to Aligned Array
alignedlist=[]
for key, val in aligned_dict.items():
		alignedlist.append(val)
alignedArray = np.array(alignedlist)
print(alignedArray)
with open('AlighnedArray.txt', 'w') as f:
	np.savetxt(f, alignedArray, delimiter = ' ', fmt='%1.8f')


# In[29]:


# Do we need to centerize data ?
def center(data):
	'''
	input: a dictionary where each value is a model in the form of a flattened array, and each array contains the coordinates of the atoms of that model.

	Method:
		Constructs an m by n array where m is the total number of coorniates of all atoms (e.g., for 1ail with 70 atoms,  m = 70 * 3 = 270), and n is the number of models, i.e., n= 50,000+
		subtracts the mean of the row elements from each value of the rows

	returns: the centered array, i.e., the result of the above method
	'''
#Directly taking array now after training
	#biglist = []
	#for key, val in aligned_dict.items():
	#	biglist.append(val)
	#data = np.array(biglist)
	data = data.T
	mean = data.mean(axis=1).reshape(-1, 1)
	data = data - data.mean(axis=1).reshape(-1, 1)
	return data, mean


# In[30]:


# Do we need to centerize data ? 

centered_data, mean = center(alignedArray)

#centered_data_train = np.arange(centered_data_train.shape[0]*centered_data_train.shape[1]).reshape(centered_data_train.shape[1], centered_data_train.shape[0]) 


#print(centered_data_train)


#centered_data_test = np.arange(centered_data_test.shape[0]*centered_data_test.shape[1]).reshape(centered_data_test.shape[1], centered_data_test.shape[0]) 
centered_data = centered_data.T
print(centered_data.shape)


with open('Version2centerData.txt', 'w') as f:
	np.savetxt(f, centered_data, delimiter = ' ', fmt='%1.8f')


# In[37]:


#Minmax scalled but why?
#train_test_all_scaled = minmax_scale(centered_data, axis = 0)

#print(train_test_all_scaled.shape)
#print(test_scaled.shape)

#ncol = train_scaled.shape[1]
#print(ncol)


#print(test_scaled)


#print(test_scaled.shape)
scaler = StandardScaler()
scaler.fit(centered_data)
X_scaled = scaler.transform(centered_data)
print(X_scaled.shape)
print(X_scaled)


# In[40]:


#Initial Concat of 1dtja set ( Which is aligned) and their co-ordinated energy

#initialpc_and_energy = np.concatenate((train_test_all_scaled, energyarray), axis = 1)
initialpc_and_energy = np.concatenate((X_scaled, energyarray), axis = 1)

with open('Version2Initial_center_data_energy_1dtja.txt', 'w') as f:
	np.savetxt(f, initialpc_and_energy, delimiter = ' ', fmt='%1.8f')


# In[41]:


print(initialpc_and_energy.shape)


# In[42]:


#Instead of All data, here we are just taking first 1000 data
X = initialpc_and_energy[0:1000,:]
print(X.shape)


# In[43]:


train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
print(train.shape)
print(test.shape)


# In[44]:


# separate Energy from train data 
train_after_enegy, EnergyAfterTrain = train[:, 0:222], train[:, 222:223]
print(train_after_enegy.shape)
print(EnergyAfterTrain.shape)

# separate Energy from test data
test_after_enegy, EnergyAfterTest = test[:, 0:222], test[:, 222:223]
print(test_after_enegy.shape)
print(EnergyAfterTest.shape)

print(train_after_enegy)


# In[42]:





# In[45]:


train_x = np.array(train_after_enegy)
test_x = np.array(test_after_enegy)
print(train_x.shape)
print(test_x.shape)
print(test_x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


####################### APROACH#############################
#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# input_dim = X.shape[1]
encoding_dim = 2  
input_img = Input(shape=(222,))
 


encoded = Dense(128, activation = 'linear')(input_img)
encoded = Dense(64, activation = 'softplus')(encoded)
encoded = Dense(2, activation = 'softplus')(encoded)


#encoded = Dense(128)(input_img)
#LR = LeakyReLU(alpha=0.1)(encoded)
#encoded = Dense(64)(LR)
#LR = LeakyReLU(alpha=0.1)(encoded)
#encoded = Dense(2, activation = 'softplus')(LR)



# Decoder Layers
decoded = Dense(64, activation = 'softplus')(encoded)
decoded = Dense(128, activation = 'softplus')(decoded)
decoded = Dense(222, activation = 'linear')(decoded)
#decoded = Dense(64)(encoded)
#LR = LeakyReLU(alpha=0.1)(decoded)
#decoded = Dense(128)(LR)
#LR = LeakyReLU(alpha=0.1)(decoded)
#decoded = Dense(2, activation = 'sigmoid')(LR)


# ######################
autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())


# In[148]:


#history = autoencoder.fit(train_x, train_x,
#                epochs=1000,
#                batch_size=100,
#                shuffle=True,
#                validation_split=0.1,
#                verbose = 0)
#print(history.history)   


# In[47]:


#Should we use train_x or x_train ?


history = autoencoder.fit(train_x, train_x,
                epochs=2000,
                batch_size=64,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)
				
#plot our loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()				
			


# In[ ]:





# In[99]:


# Use Encoder level to reduce dimension of train and test data
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

###### Use decoder level
# decoder_layer = autoencoder.layers[-1]
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

print(decoder.summary())


# In[100]:


#encoded_data = encoder.predict(test_scaled)
encoded_data = encoder.predict(test_x)
print(encoded_data.shape)
print(encoded_data)


# In[ ]:






# In[101]:


auto_data=autoencoder.predict(test_x)
print(auto_data.shape)
print(test_x-auto_data)
with open('Version2DifferenceData.txt', 'w') as f:
	np.savetxt(f, test_x-auto_data, delimiter = ' ', fmt='%1.8f')


# In[56]:


#Predict the new train and test data using Encoder
#encoded_train = encoder.predict(train_x)


#encoded_test = (encoder.predict(test_x))


# Plot
#plt.plot(encoded_train[:,:])
#plt.show()

# Plot
#plt.plot(encoded_test[:,:])
#plt.show()


# In[102]:


#Concat auto encoder result with enegy
encoder_and_energy = np.concatenate((encoded_data, EnergyAfterTest), axis = 1)

with open('Version2AutoEncoder_energy_1dtja.txt', 'w') as f:
	np.savetxt(f, encoder_and_energy, delimiter = ' ', fmt='%1.8f')


# In[205]:


ae1= np.array(encoder_and_energy[:,0:1])
ae2= np.array(encoder_and_energy[:,1:2])
eneryscale=np.array(encoder_and_energy[:,2:3])
cm = plt.cm.get_cmap('RdYlBu')

plt.scatter(x=ae1,y=ae2,c=eneryscale,s=110, cmap=cm)
plt.title("AE1 & AE2 for energy scale")
plt.xlabel("AE1")
plt.ylabel("AE2")
cbar= plt.colorbar()
cbar.ax.invert_yaxis()
#cbar.ax.set_yticklabels("Eneryscale", labelpad=+1)
#cbar.set_label("Eneryscale", labelpad=+1)
#cbar.ax.invert_yaxis()

plt.show()


# In[158]:


originalD=test_x
print(originalD.shape)
#print(originalD[0])
reconstructedD=auto_data
print(reconstructedD.shape)

Eu=euclidean_distances(originalD,reconstructedD )
euD= Eu/14.9
#print(Eu)
print(euD.shape)

Result=0
FinalResult=0
Sumresult=0
ary = [];
for i in range(400):
    for j in range(222):
        Result=0
        Result=np.sqrt(np.sum((originalD[i][j] - reconstructedD[i][j])**2))
        Sumresult= Sumresult+Result
        print(i)
        print(Sumresult)
    FinalResult= Sumresult/14.9
ary.append(FinalResult)    
    #Sumresult[i]=Result

#print(ary.shape)


# In[165]:


a=np.array(originalD)
b=np.array(reconstructedD)
c=[]

for i in range(400):
	sum=0
	for j in range(2):
		sum+=(a[i][j]-b[i][j])**2
	c.append(math.sqrt(sum)/math.sqrt(222))

c=np.array(c)
print(c)


# In[194]:


x = c
num_bins = 20
n, bins, patches = plt.hist(x, num_bins,facecolor='blue', alpha=0.5,edgecolor='black', linewidth=1.2)
plt.ylabel('Root Mean Squared Distance');

plt.grid(True);
plt.title("Without Normalization")
plt.show()

x = c
num_bins = 20
n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5,edgecolor='black', linewidth=1.2)
plt.ylabel('Root Mean Squared Distance');
plt.grid(True);
plt.title("With Normalization")
plt.show()


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# In[141]:


#A=np.array([1,2],[1,1])
a = np.array(
    [[5,6,5], # day 1
     [5,5,5]])
b = np.array(
    [[4,4,4], # day 1
     [3,3,3]])
#B=np.array([1,1],[1,1])
s = np.sum((a-b)**2)
print(a[1])


# In[195]:


####################### APROACH#############################
#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# input_dim = X.shape[1]
encoding_dim = 2  
input_img = Input(shape=(222,))
 


encoded = Dense(128, activation = 'linear')(input_img)
encoded = Dense(64, activation = 'softplus')(encoded)
encoded = Dense(2, activation = 'softplus')(encoded)


#encoded = Dense(128)(input_img)
#LR = LeakyReLU(alpha=0.1)(encoded)
#encoded = Dense(64)(LR)
#LR = LeakyReLU(alpha=0.1)(encoded)
#encoded = Dense(2, activation = 'softplus')(LR)



# Decoder Layers
decoded = Dense(64, activation = 'softplus')(encoded)
decoded = Dense(128, activation = 'softplus')(decoded)
decoded = Dense(222, activation = 'sigmoid')(decoded)
#decoded = Dense(64)(encoded)
#LR = LeakyReLU(alpha=0.1)(decoded)
#decoded = Dense(128)(LR)
#LR = LeakyReLU(alpha=0.1)(decoded)
#decoded = Dense(2, activation = 'sigmoid')(LR)


# ######################
autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())


# In[196]:


#Should we use train_x or x_train ?


history1 = autoencoder.fit(train_x, train_x,
                epochs=2000,
                batch_size=64,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)
				
#plot our loss 
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[ ]:




