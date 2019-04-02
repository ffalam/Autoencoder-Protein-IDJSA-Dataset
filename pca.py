import numpy as np
import xml.etree.ElementTree as ET
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
#sys.path.insert(0, '/home/localadmin/sbl/codes/general_codes')
#import utils
#reload(utils)
#from utils import align


EPSILON = 1.0 * (2**-36)



def align(coordinate_file):
	'''
	Input: File contains lines, where each line contains the coordinates of a model, e.g., if model 1 has 70 atoms, each with 3 coordinates  (3*70 = 210 coordinates),
	then the line corresponding model 1 is like this:  210 x1 y1 z1 x2 y2 z2 ... x70 y70 z70

	Alignes all the model with the first model in the cordinate_file.

	Returns: a dictionary of aligned models. Each model, i.e., each entry (value) in the dictionary is a flattened numpy array.

	NOTE: For my leader cluster codes, do not flatten the arrays.
	'''

	modelDict = {}
	ind = 0
	ref = []
	sup = SVDSuperimposer()
	with open(coordinate_file) as f:
		for line in f:
			if ind == 0:
				l = [float(t) for t in line.split()]
				l = l[1:]  # 1ail:l[1:211]
				samples = [l[i:i+3] for i in range(0, len(l), 3)]
				ref = array(samples, 'f')

				modelDict[ind] = np.ravel(ref)
				ind += 1
			else:
				l = [float(t) for t in line.split()]
				l = l[1:]  # 1ail:l[1:211]
				samples = [l[i:i+3] for i in range(0, len(l), 3)]
				seq = array(samples, 'f')
				s = sup.set(ref, seq)
				sup.run()
				z = sup.get_transformed()
				modelDict[ind] = np.ravel(z)
				ind += 1
	return modelDict, ref




def center(aligned_dict):
	'''
	input: a dictionary where each value is a model in the form of a flattened array, and each array contains the coordinates of the atoms of that model.

	Method:
		Constructs an m by n array where m is the total number of coorniates of all atoms (e.g., for 1ail with 70 atoms,  m = 70 * 3 = 270), and n is the number of models, i.e., n= 50,000+
		subtracts the mean of the row elements from each value of the rows

	returns: the centered array, i.e., the result of the above method
	'''

	biglist = []
	for key, val in aligned_dict.items():
		biglist.append(val)
	data = np.array(biglist)
	data = data.T
	mean = data.mean(axis=1).reshape(-1, 1)
	data = data - data.mean(axis=1).reshape(-1, 1)
	return data, mean



def svd(data):
	'''
	Input: Centerd or scaled data of size m by n. n = number of models, m = total number of coordinates in each model.
	Method:
		1. prepare the data: divides each element by sqrt(n-1)
		2. run sklearn's PCA on data
		3. extract the singular values, square them to get the variances
		4. extract the eigenvectors/principle_components
		5. Compute the projected data, i.e., original data projected to the priciple components
	OUtputs:
		1. squared singular values (variances)
		2. principle components
		3. projected data by using sklearn pca's transform()
		4. projected data by multiplying the eigenvectors/principle_components with the data
	'''
	original_data = data
	# prepare the data
	data = data.T
	Y = np.divide(data, math.sqrt((data.shape[1]-1)))
	pca = PCA()
	pca.fit(Y)
	pcs = pca.components_
	explained_var = pca.explained_variance_
	singular_values = pca.singular_values_
	singular_values = np.square(singular_values) # eigenvectors/PCs
	projected_data_1 = pca.transform(data)
	projected_data_1 = projected_data_1.T

	# multiply the principle components with the data to generate the projected data
	pcs = pcs.T
	idx = np.argsort(singular_values)[::-1]
	singular_values = singular_values[idx]
	pcs = pcs[:, idx]

	projected_data_2 = np.dot(pcs.T, original_data)

	return singular_values, pcs, projected_data_1, projected_data_2



def Eig(data):
	'''
	Input: Centerd or scaled data of size m by n. n = number of models, m = total number of coordinates in each model.
	Method:
		1. prepare the data: divides each element by sqrt(n-1)
		2. calculate the covariance matrix, which is a square matrix. Numpy's linalg.eig requires a square matrix
		3. run numpy's linalg.eig on data
		4. collect the eigenvalues and eigenvectors
		5. Compute the projected data by multiplying the eigenvectors/principle_components with the data
	OUtputs:
		1. eigenvalues
		2. eigenvectors
		3. projected data
	'''
	#calculate the covariance matrix, which is a square matrix. Numpy's linalg.eig requires a square matrix
	n_minus_1 = data.shape[1]-1
	mult = np.dot(data, np.transpose(data))
	cov = np.divide(mult, n_minus_1)
	eigenvals, eigenvecs = np.linalg.eig(cov)
	idx = np.argsort(eigenvals)[::-1]
	eigenvals = eigenvals[idx]
	eigenvecs = eigenvecs[:, idx]
	eigenvecs_all = eigenvecs

	projected_data_eig1 = np.dot(eigenvecs.T, data)

	eigenvecs = eigenvecs[:, :2]    # retain only top two pcs
	projected_data_eig2 = np.dot(eigenvecs.T, data)
	return eigenvals, eigenvecs_all, projected_data_eig1, projected_data_eig2


def accSum(eigenvals):
	'''
	Input: eigenvalues as a 1D array or list
	Method: Divides each eigenvalue by the highest value in the list/array, then multiplies each of them with 100 to generate percentage of accumulated variances
	Output: returns the accumulated variances
	'''
	non_zero_eigvals = eigenvals[ 0 < eigenvals ]
	accSum = np.cumsum(non_zero_eigvals)
	accSum /= accSum[ accSum.shape[0] - 1 ]
	accSum *= 100
	return accSum



def main():
	'''
	Requires the location of the coordinate file.
	Each file contains lines, where each line contains the coordinates of a model, e.g., if model 1 has 70 atoms, each with 3 coordinates  (3*70 = 210 coordinates),
	then the line corresponding model 1 is like this:  210 x1 y1 z1 x2 y2 z2 ... x70 y70 z70

	'''
	coordinate_file = 'small.txt'



	# align the models with the first one in the file. No need to call this function if already pickled to disk.

	aligned_dict, ref = align(coordinate_file)


	# center the data by subtracting the mean of the rows from each row element. No need to do it again if centered data is already pickled to disk.
	centered_data, mean = center(aligned_dict)
	print (centered_data.shape)


	# perform svd via sklearn's PCA
	squared_singulars, pcs, projected_data_1, projected_data_2 = svd(centered_data)

	accSum_svd = accSum(squared_singulars)
	print ('accSum from svd: ')
	print (accSum_svd[:10])
	print()

	# compute the eigenvalues and eigenvectors using numpy's linalg.eig
	eigenvalues, eigenvecs, projected_data_eig1, projected_data_eig2 = Eig(centered_data)
	accSum_eig = accSum(eigenvalues)
	print ('accSum from eig: ')
	print (accSum_eig[:10])


	eigenvalues = eigenvalues.reshape(-1,1)

	accSum_eig = np.array(accSum_eig).reshape(-1,1)
	indices = np.array(list(range(1, len(accSum_eig)+1))).reshape(-1,1)
	write_to_file = np.concatenate((indices, accSum_eig), axis = 1)
	ref = np.ravel(ref).reshape(-1,1)
	ref_mean_eig_pcs = np.concatenate((ref, mean, eigenvalues, eigenvecs), axis = 1)


	projected_data_eig2 = projected_data_eig2.T

	energylist = []
	#read the energy file
	with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)

	#energylist = energylist[:-1]  # only for 1aly
	energyarray = np.array(energylist).reshape(-1,1)

	pc_and_energy = np.concatenate((projected_data_eig2, energyarray), axis = 1)

	with open('pc1_pc2_energy_1dtja.txt', 'w') as f:
		np.savetxt(f, pc_and_energy, delimiter = ' ', fmt='%1.8f')



if __name__ == '__main__':
	main()
