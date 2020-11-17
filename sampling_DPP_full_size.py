# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:07:07 2020

@author: yadav
"""

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
import scipy
import contour_integral
import matplotlib    #pylab is submodule in matplotlib
import random
import timeit
import sys
import decimal
from decimal import Decimal
from scipy.stats import poisson

theta = 1.0001
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
gamma = 3.0
beta = 1.0
iteration = 13.0


data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_gamma=3.0_theta=1.0001_18000points_iter13.0.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_method_2_epsi=1e-4_gamma=3.0_theta=1.0001_rho=2.0_18000points_corrected4_iter13.0.txt",float)

f_out1=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta="+str(theta)+"_gamma="+str(gamma)+"_1800points_1.txt","w")
f_out2=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_independent_1800points_1.txt","w")
f_out3=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta="+str(theta)+"_gamma="+str(gamma)+"_1800points_2.txt","w")
f_out4=file("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_independent_1800points_2.txt","w")

x = data1[:,0]
#x = x.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
#x.astype(numpy.float64)
x_short = x[::10]    # Slicing an arrray a, a[start:stop:step]. See https://stackoverflow.com/questions/25876640/subsampling-every-nth-entry-in-a-numpy-array 

sigma = data2[:,1]
sigma = sigma.reshape(-1, 1)    #reshape function coverts array of shape (n,) to (n,1). See https://stackoverflow.com/questions/39549331/reshape-numpy-n-vector-to-n-1-vector
sigma=sigma[::-1]    # to reverse the order of the elements of array. See https://www.askpython.com/python/array/reverse-an-array-in-python
sigma_short = sigma[::10]    # Slicing an arrray a, a[start:stop:step]. See https://stackoverflow.com/questions/25876640/subsampling-every-nth-entry-in-a-numpy-array 

R = numpy.empty([len(data1[:,0])/10+1,len(data1[:,0])/10+1],float)

#u = numpy.empty([len(x),len(x)],float)
u = numpy.empty([len(x)/10+1,len(x)/10+1],float)

#u_delta = numpy.empty([len(x),len(x)],float)
u_delta = numpy.empty([len(x)/10+1,len(x)/10+1],float)

delta_x = (x[len(x)-1]-x[0])/(len(x)-1)    # delta is constant and is average spacing between points on support.    
epsi=1e-319*delta_x    #1e-319 is smallest fraction till we get math error in python 2.7

#for i in range(len(x)):    # function len() on array gives no. of rows of array
for i in range(0,len(x),10):    # function len() on array gives no. of rows of array
    
#    for j in range(len(x)):    # function len() on array gives no. of rows of array
    for j in range(0,len(x),10):    # function len() on array gives no. of rows of array
#        print(i,j)
        if(i==j):
#            u[i,j] = sys.float_info.max    # maximum possible float value in python. See https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python
#            u[i/100,j/100] = sys.float_info.max    # maximum possible float value in python. See https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python
#            u[i/100,j/100] = 10000000    # maximum possible float value in python. See https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python
#            u[i/100,j/100] = -math.log(abs(x[i]-(x[j]+epsi)))-gamma*math.log(abs(x[i]**theta-(x[j]+epsi)**theta))
#            u[i/100,j/100] = -math.log(abs(Decimal(x[i])-(Decimal(x[j])+Decimal(epsi))))-gamma*math.log(abs(Decimal(x[i]**theta)-(Decimal(x[j])+Decimal(epsi**theta))))    #Python has the same limitations for floating point arithmetic as all the other languages. You can use Decimal to get the accurate result. See https://stackoverflow.com/questions/36541776/floating-point-subtraction-in-python
            u[i/10,j/10] = -math.log(abs(epsi))

        else:
#            u[i,j] = -math.log(abs(x[i]-x[j]))-gamma*math.log(abs(x[i]**theta-x[j]**theta))
            u[i/10,j/10] = -math.log(abs(x[i]-x[j]))-gamma*math.log(abs(x[i]**theta-x[j]**theta))
        
#        if(i==len(x)-1):
#            delta_x = x[i]-x[i-1]
#        else:
#            delta_x = x[i+1]-x[i]
            
#        delta_x = (x[len(x)-1]-x[0])/(len(x)-1)    # delta is constant and is average spacing between points on support.    
#        u_delta[i,j] = delta_x*u[i,j]
        u_delta[i/10,j/10] = delta_x*u[i/10,j/10]

#u_delta = u_delta-numpy.amin(u_delta)    # To make make minimum of u_delta (which is usually negative) equal to zero. This shifts all the entries of u_delta by constant amount so that they are non-negative. 

print('Matrix u_delta has been computed') 

R = (1.0/beta)*numpy.linalg.inv(u_delta)    # See eq.(50) in review 'random matrix theory of quantum transport' by Beenakker

print('Two point correlation function R has been computed') 

#matplotlib.pyplot.imshow(R, vmin=numpy.amax(R)/100, vmax=numpy.amax(R)/10)    # see https://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib
#matplotlib.pyplot.imshow(R, vmin=numpy.amin(R), vmax=numpy.amax(R)/50)    # see https://stackoverflow.com/questions/16492830/colorplot-of-2d-array-matplotlib
#matplotlib.pylab.show()

#K = (numpy.dot(sigma,sigma.T)+R)**(0.5)    # See eq.(50) and eq.(43) in review 'random matrix theory of quantum transport' by Beenakker
#K = (numpy.dot(sigma_short,sigma_short.T)+R)**(0.5)    # See eq.(50) and eq.(43) in review 'random matrix theory of quantum transport' by Beenakker
K=scipy.linalg.sqrtm(numpy.dot(sigma_short,sigma_short.T)+R)

print('Kernel K has been computed')

Lambda_K, v_K = numpy.linalg.eig(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 
#Lambda_K, v_K = numpy.linalg.eigh(K)    # to find out eigenvalues and eigenvectors of real symmetric matrix. 

#Lambda_L = Lambda_K/(Lambda_K-1.0)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
Lambda_L = Lambda_K/(1.0-Lambda_K)    # relation between eigenvalues of kernel K and eigenvalues of kernel L.
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
Y_indp = numpy.empty(N0,int)

#XX = numpy.zeros(N0,int)
#XX_indp = numpy.zeros(N0,int)
#YY = numpy.zeros(N0,int)
#YY_indp = numpy.zeros(N0,int)

XX = numpy.zeros(N,int)
XX_indp = numpy.zeros(N,int)
YY = numpy.zeros(N,int)
YY_indp = numpy.zeros(N,int)



print('Loop 1 of sampling algorithm has been computed') 
print('size of the sample is',N0)

# Following is the loop 2 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 
#N0=20

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
            XX[j]=Y[j]
            f_out1.write(str(Y[j])+'\n')

            for l in range(N0):
                v_L[l] = v_L[l]-(numpy.dot(v_L[l],e_m_prime))*e_m_prime
            break        

for k in range(N0):
    
    Y_indp[k] = random.randint(0,N-1)    # generates random integer from 0 to N-1
    XX_indp[k]=Y_indp[k]
    f_out2.write(str(Y_indp[k])+'\n')


#numpy.histogram(Y)
#matplotlib.pyplot.hist(Y, bins=100)
#matplotlib.pylab.show()

#matplotlib.pyplot.hist(Y_indp, bins=100)
#matplotlib.pylab.show()


f_out1.close()    # () at the end is necessary to close the file 
f_out2.close()    # () at the end is necessary to close the file             
            
#numpy.savetxt("input_output_files/theta_"+str(theta)+"/V_eff/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_c="+str(c_short_delta)+"_theta="+str(theta)+"_gamma="+str(gamma)+"_9000points.txt", Y, newline='n')
 
# Following is the loop 1 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 


J_2 = numpy.zeros(N,int)

for i in range(N):    # function len() on array gives no. of rows of array
    p = random.random()    # generates random floating number from [0.0,1.0)
    if((abs(Lambda_L[i])/(1.0+abs(Lambda_L[i])))>=p):
        J_2[i]=i

J_2_masked = numpy.ma.masked_equal(J_2,0)    # to mask all the zeros of array J
J_2 = J_2_masked[~J_2_masked.mask]    # to remove all the masked values of array J_masked

N0 = len(J_2)
Y_2 = numpy.empty(N0,int)
Y_2_indp = numpy.empty(N0,int)


print('Loop 1 of sampling algorithm has been computed') 
print('size of the sample is',N0)

# Following is the loop 2 for sampling of a DPP algorithm 1 in Kulesza and Taskar. See notes. 
#N0=20

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
            volume = (numpy.dot(v_L[J_2[k]].T,e_m_prime))**2.0    # .T denotes transpose
            summation = summation + volume
        if((1.0/mod_V)*summation>=p_prime):
            Y_2[j]=m_prime
            YY[j]=Y_2[j]            
            f_out3.write(str(Y_2[j])+'\n')

            for l in range(N0):
                v_L[l] = v_L[l]-(numpy.dot(v_L[l],e_m_prime))*e_m_prime
            break        

for k in range(N0):
    
    Y_2_indp[k] = random.randint(0,N-1)    # generates random integer from 0 to N-1
    YY_indp[k]=Y_2_indp[k]
    f_out4.write(str(Y_2_indp[k])+'\n')

f_out3.close()    # () at the end is necessary to close the file 
f_out4.close()    # () at the end is necessary to close the file             

#XX_poisson = 10*poisson.rvs(90, size=N0*10)
#YY_poisson = 10*poisson.rvs(90, size=N0*10)

#XX_poisson = poisson.rvs(90, size=N0)
#YY_poisson = 10*poisson.rvs(90, size=N0)
 
#matplotlib.pylab.plot(XX,YY)
#matplotlib.pylab.scatter(XX,YY, s=5, color='red')
#matplotlib.pylab.show()

#matplotlib.pylab.plot(XX_indp,YY_indp)
#matplotlib.pylab.scatter(XX_indp,YY_indp, s=5, color='green')
#matplotlib.pylab.show()

#matplotlib.pylab.scatter(XX_poisson,YY_poisson, s=5, color='blue')
#matplotlib.pylab.xlim(0,180)
#matplotlib.pylab.ylim(0,180)
#matplotlib.pylab.show()

#plt.scatter 
#r = poisson.rvs(1, size=1000)
       