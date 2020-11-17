# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:06:28 2020

@author: yadav
"""

# code to plot nu
import math
import cmath
import numpy
import matplotlib    #pylab is submodule in matplotlib
#import matplotlib.pylab 
#import pylab    # Use this step to run in anaconda

theta = 1.0001
rho = 2.0    # V(x)=rho*x. for \theta=1 and V(x)=rho*x, V_eff(x)=2*rho*x/(1+gamma)=rho_eff*x (See our beta ensembles paper)
gamma = 3.0
beta = 1.0
iteration = 13.0

data1 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta=1.0001_gamma=3.0_1800points_1.txt",float)
data2 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_independent_1800points_1.txt",float)
data3 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_sampling_DPP_theta=1.0001_gamma=3.0_1800points_2.txt",float)
data4 = numpy.loadtxt("input_output_files/theta_"+str(theta)+"/linear_pot/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/Y_independent_1800points_2.txt",float)


XX = data1    # The plain [:] operator slices from beginning to end-1, Eg. a[:,0:n] gives all the rows from 0th column to n-1 column 
YY = data3
#y1 = data1[:,1]*10**10
XX_indp = data2    # The plain [:] operator slices from beginning to end-1, Eg. a[:,0:n] gives all the rows from 0th column to n-1 column 
YY_indp = data4

shape_diff = numpy.array(YY.shape) - numpy.array(XX.shape)
XX = numpy.lib.pad(XX, ((0,shape_diff[0])), 'constant', constant_values=(0))    # To pad zeros around given array. See https://stackoverflow.com/questions/30229104/python-increase-array-size-and-initialize-new-elements-to-zero
XX_indp = numpy.lib.pad(XX_indp, ((0,shape_diff[0])), 'constant', constant_values=(0))    # To pad zeros around given array. See https://stackoverflow.com/questions/30229104/python-increase-array-size-and-initialize-new-elements-to-zero


matplotlib.pylab.scatter(XX,YY, s=2, color='blue', label="DPP")
matplotlib.pylab.xlim(600,1400)
matplotlib.pylab.ylim(600,1400)
matplotlib.pylab.show()

matplotlib.pylab.scatter(XX_indp,YY_indp, s=2, color='blue',label="Poisson")
matplotlib.pylab.xlim(600,1400)
matplotlib.pylab.ylim(600,1400)
matplotlib.pylab.show()

matplotlib.pyplot.hist(XX, bins=1000)
matplotlib.pylab.show()

matplotlib.pyplot.hist(YY, bins=1000)
matplotlib.pylab.show()

matplotlib.pyplot.hist(XX_indp, bins=1000)
matplotlib.pylab.show()

matplotlib.pyplot.hist(YY_indp, bins=1000)
matplotlib.pylab.show()



