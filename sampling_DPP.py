# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:34:22 2020

@author: yadav
"""

# code to generate two point correlation function, Kernels K,L and sampling from DPP (see section 2.4 page number 145) in review 'random matrix theory of quantum transport' by Beenakker
# The code uses eq.(50) in review 'random matrix theory of quantum transport' by Beenakker to compute two point correlation function.

import math
import cmath
import numpy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib
import random
import timeit
import sys

theta = 1.0001
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
gamma = 0.4
beta = 1.0
iteration = 39.0

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_gamma="+str(gamma)+"_theta="+str(theta)+"_18000points_iter"+str(iteration)+".txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_method_2_epsi=1e-4_gamma="+str(gamma)+"_theta="+str(theta)+"_rho="+str(rho)+"_18000points_corrected4_iter"+str(iteration)+".txt",float)

f_out=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta="+str(theta)+"_gamma="+str(gamma)+"_18000points.txt","w")

x = data1[:,0]

sigma = data2[:,1]
sigma = sigma.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
sigma=sigma[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python


u = numpy.empty([len(x),len(x)],float)
u_delta = numpy.empty([len(x),len(x)],float)

for i in range(len(x)):    # function len() on array gives no. of rows of array
    
    for j in range(len(x)):    # function len() on array gives no. of rows of array
        
        if(i==j):
            u[i,j] = sys.float_info.max    # maximum possible float value in python. See https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python
        else:
            u[i,j] = -math.log(abs(x[i]-x[j]))-gamma*math.log(abs(x[i]**theta-x[j]**theta))
        
        if(i==len(x)-1):
            delta_x = x[i]-x[i-1]
        else:
            delta_x = x[i+1]-x[i]
        u_delta[i,j] = delta_x*u[i,j]

print('Matrix u_delta has been computed') 

R = (1.0/beta)*numpy.linalg.inv(u_delta)    # See eq.(50) in review 'random matrix theory of quantum transport' by Beenakker

print('Two point correlation function R has been computed') 

K = (numpy.dot(sigma,sigma.T)+R)**(0.5)    # See eq.(50) and eq.(43) in review 'random matrix theory of quantum transport' by Beenakker

print('Kernel K has been computed')

Lambda_K, v_K = numpy.linalg.eig(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 
#Lambda_K, v_K = numpy.linalg.eigh(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 

Lambda_L = Lambda_K/(Lambda_K-1.0)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
v_L=v_K    # eigenvectors of kernel K and kernel L are same.

print('eigenvalues and eigenvectors of Kernel L has been computed') 

N = len(K)   

J = numpy.zeros(N,int)

# Following is the loop 1 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 

for i in range(N):    # function len() on array gives no. of rows of array
    p = random.random()    # generates random floating number from [0.0,1.0)
    if((abs(Lambda_L[i])/(1.0+abs(Lambda_L[i])))>=p):
        J[i]=i

J_masked = numpy.ma.masked_equal(J,0)    # to mask all the zeros of array J
J = J_masked[~J_masked.mask]    # to remove all the masked values of array J_masked

N0 = len(J)
Y = numpy.empty(N0,int)

print('Loop 1 of sampling algorithm has been computed') 
print('size of the sample is',N0)

# Following is the loop 2 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 

for j in range(N0):
    mod_V = N0-j
    print('iterations remaining are',N0-j)
    while True:
        m_prime = random.randint(0,N-1)    # generates random integer from 0 to N-1
#        m_prime = random.randrange(N)    # generates random integer from 0 to N-1
        e_m_prime = numpy.zeros(N,float)
        e_m_prime[m_prime] = 1.0
        p_prime = random.random()    # generates random floating number from [0.0,1.0)
        summation = 0
        for k in range(N0):
            volume = (numpy.dot(v_L[J[k]].T,e_m_prime))**2.0    # .T denotes transpose
            summation = summation + volume
        if((1.0/mod_V)*summation>=p_prime):
            Y[j]=m_prime
            f_out.write(str(Y[j])+'\n')

            for l in range(N0):
                v_L[l] = v_L[l]-(numpy.dot(v_L[l],e_m_prime))*e_m_prime
            break        

f_out.close()    # () at the end is necessary to close the file             
#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt", Y, newline='n')
 
 
        