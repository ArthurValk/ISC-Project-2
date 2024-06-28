# -*- coding: utf-8 -*-
"""
Code by Tristan van Leeuwen, modified by Arthur Valk & Senne Versteeg to
show emergence of multicolinearity in some matrices A.
"""

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
from datetime import datetime
import os
import pandas as pd

def scanlines(theta,nd):
    # function to generate scanlines for given nd and theta, which can then
    # be used in scanline_image
    ndlist = np.zeros(nd+1)
    detectorlist = np.zeros(nd)
    for i in range(0,nd):
        ndlist[i] = -1 + i*2/nd
        detectorlist[i] = -1 + (2*i+1)/nd
    ndlist[nd] = 1
    result = np.zeros((nd*len(theta),4))
    for j in range(0,len(theta)):
        angle = theta[j]
        rc = -1/np.tan(angle)
        for i in range(0,nd):
            point = np.array([np.cos(angle)*(-1 + (2*i+1)/nd)
                              ,np.sin(angle)*(-1 + (2*i+1)/nd)])
            result[nd*j+i][0] = rc
            result[nd*j+i][1] = -rc*point[0]+point[1]
            result[nd*j+i][2] = detectorlist[i]
            result[nd*j+i][3] = angle + np.pi/2
    return result
        
def scanline_image(lines,n=2,legend = False):
    # code by Tristan van Leeuwen, modified by Arthur Valk & Senne Versteeg
    # to generate visualisation of scanlines for given coefficients
    # and gridsize
    # creating pixelgrid
    x = np.array([-1+2/n*i for i in range(0,n+1)])
    y = np.array([-1+2/n*i for i in range(0,n+1)])
    
    # plotting pixelgrid
    xx,yy = np.meshgrid(x,y)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(4,4)
    
    ax.plot(xx,yy,'b')
    ax.plot(yy,xx,'b')
    
    # creating and plotting scanlines
    for j in range(0,len(lines)):
        i = lines[j]
        a,b = i[0],i[1]
        x_plus = (1.5-b)/a
        x_minus = -(1.5+b)/a
        if a <= 0:
            xmin = max(x_plus,-1.5)
            xmax = min(1.5,x_minus)
        if a > 0:
            xmin = max(x_minus,-1.5)
            xmax = min(1.5,x_plus)
        xarray = np.linspace(xmin,xmax)
        yarray = a*xarray + b
        ax.plot(xarray,yarray,label = f"{j+1}")
    
    ax.set_aspect(1)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    if legend == True:
        ax.legend()
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\scanlines {now}"
    plt.savefig(filepath)

# Constructing matrix A as in lecture 3, code courtesy of Tristan van Leeuwen.

def ray(n, s, theta):
    """
    Trace ray:
        x(t) =  t*sin(theta) + s*cos(theta)
        y(t) = -t*cos(theta) + s*sin(theta)
    through volume [-1,1]^2, discretised with n x n pixels.
    
    Returns linear indices of intersected pixels and corresponding intersection lengths
    """
    
    # define boundaries of pixels
    h = 2/n
    x = -1 + np.linspace(0,n,n+1)*h
    y = -1 + np.linspace(0,n,n+1)*h
    
    # compute all intersections with horizontal and vertical grid lines
    # in terms of path length parameter t
    t = []
    if np.abs(np.sin(theta)) > 0:
        tx = (x - s*np.cos(theta))/np.sin(theta)
        t = np.concatenate((t,tx))
    if np.abs(np.cos(theta)) > 0 :
        ty = -(y - s*np.sin(theta))/np.cos(theta)
        t = np.concatenate((t,ty))
    
    # sort t in increasing order
    t = np.sort(t[np.isfinite(t)])
    
    # now trace the ray and store pixel-indices and path lengths
    nt = len(t)
    I = []
    w = []
    for i in range(nt-1):
        # entry point of ray
        x1 = t[i]*np.sin(theta) + s*np.cos(theta)
        y1 = -t[i]*np.cos(theta) + s*np.sin(theta)
        
        # exit point of ray
        x2 = t[i+1]*np.sin(theta) + s*np.cos(theta)
        y2 = -t[i+1]*np.cos(theta) + s*np.sin(theta)
        
        # pixel indices
        ix = int(((x1 + x2)/2+1)//h)
        iy = int(((y1 + y2)/2+1)//h)
        
        # only take pixels in volume [-1,1]^2 in to account
        if (0 <= ix < n) and (0 <= iy < n):
            I.append(int(iy + ix*n))
            w.append(t[i+1] - t[i])
        
    return I,w

def Amatrix(n,theta,nd,generateimage = False,legend = False):
    # function to generate matrix A from output of ray function
    lines = scanlines(theta,nd)
    A = np.zeros((nd*len(theta),n*n))
    for i in range(0,len(lines)):
        rayoutput = ray(n,lines[i][2],lines[i][3])
        for j in range(0,len(rayoutput[0])):
            A[i][rayoutput[0][j]] = rayoutput[1][j]
    if generateimage == True:
        scanline_image(lines,n=n,legend = legend)
    return A

def compress(u,factor):
    # function to compress image by some factor
    size = np.shape(u)[0]
    newsize = size//factor
    cmatrix = np.zeros((newsize,size))
    for j in range(0,size):
        cmatrix[j//factor][j] = 1
    # for j in cmatrix:
        # print(j)
    result = (cmatrix @ u @ np.transpose(cmatrix))/factor**2
    return result

def OLS_forward_error_test(factor,ntheta,nd):
    # function to check what errors are created by reconstructing the
    # image using OLS
    u = shepp_logan_phantom()
    uc = compress(u,factor)
    uc_ravel = uc.ravel()
    n = np.shape(u)[0]//factor
    combinations = np.array(np.meshgrid(nd,ntheta)).T.reshape(-1, 2)
    result = np.zeros((len(combinations),3))
    # first column stores nd
    # second column stores ntheta
    # third column stores errors
    for i in range(0,len(combinations)):
        detectorcells = combinations[i][0]
        thetacount = combinations[i][1]
        theta = np.linspace(0.0001,np.pi-0.0001,thetacount)
        radontheta = 180*theta/np.pi
        A = Amatrix(n,theta,detectorcells)
        f_A = A @ uc_ravel
        f_A_radon = f_A.reshape(detectorcells,thetacount)
        u_iradon = iradon(f_A_radon,radontheta,output_size=n)
        result[i][0] = detectorcells
        result[i][1] = thetacount
        np.shape(u_iradon)
        result[i][2] = np.linalg.norm(u_iradon.ravel()-uc_ravel)
            
    # saving data
    if not os.path.isdir('Data'):
        os.mkdir("Data")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Data\ols forward data {now}.csv"
    df = pd.DataFrame(result)
    df.to_csv(filepath,header = False, index = False)
    return result,filepath

def OLS_forward_error_plot(filepath):
    df = pd.read_csv(filepath,header=None)
    dataset = np.transpose(df.values)
    # set up the figure and Axes
    fig = plt.figure(figsize=(6,6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    bottom = np.zeros_like(dataset[2])
    width = depth = 8
    
    ax1.bar3d(dataset[0], dataset[1], bottom, width, depth, dataset[2], shade=True)
    ax1.set_xlabel(r'$n_d$')
    ax1.set_ylabel(r'$n_\theta$')
    ax1.set_zlabel(r'$error$')
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\OLS forward error {now}"
    plt.savefig(filepath,dpi=300,bbox_inches = 'tight')

def OLS_reconstruction_error_test(factor,ntheta,nd):
    # function to check what errors are created by reconstructing the
    # image using OLS when compared to the inverse radon transform provided by
    # python
    u = shepp_logan_phantom()
    uc = compress(u,factor)
    uc_ravel = uc.ravel()
    n = np.shape(u)[0]//factor
    combinations = np.array(np.meshgrid(nd,ntheta)).T.reshape(-1, 2)
    result = np.zeros((len(combinations),3))
    # first column stores nd
    # second column stores ntheta
    # third column stores errors
    for i in range(0,len(combinations)):
        detectorcells = combinations[i][0]
        thetacount = combinations[i][1]
        theta = np.linspace(0.0001,np.pi-0.0001,thetacount)
        radontheta = 180*theta/np.pi
        A = Amatrix(n,theta,detectorcells)
        AA = np.transpose(A) @ A
        f_A = A @ uc_ravel
        f_A_radon = f_A.reshape(detectorcells,thetacount)
        u_iradon = iradon(f_A_radon,radontheta,output_size=n)
        u_ols = np.linalg.inv(AA) @ np.transpose(A) @ f_A
        result[i][0] = detectorcells
        result[i][1] = thetacount
        np.shape(u_iradon)
        result[i][2] = np.linalg.norm(u_iradon.ravel()-u_ols)
            
    # saving data
    if not os.path.isdir('Data'):
        os.mkdir("Data")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Data\ols data {now}.csv"
    df = pd.DataFrame(result)
    df.to_csv(filepath,header = False, index = False)
    return result,filepath
    
def OLS_reconstruction_error_plot(filepath):
    df = pd.read_csv(filepath,header=None)
    dataset = np.transpose(df.values)
    # set up the figure and Axes
    fig = plt.figure(figsize=(6,6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    bottom = np.zeros_like(dataset[2])
    width = depth = 8
    
    ax1.bar3d(dataset[0], dataset[1], bottom, width, depth, np.log(dataset[2]), shade=True)
    ax1.set_xlabel(r'$n_d$')
    ax1.set_ylabel(r'$n_\theta$')
    ax1.set_title(r'$\log(error)$')
    
    # plotting without (20,20)
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    dataset[2][0] = 0
    bottom = np.zeros_like(dataset[2])
    width = depth = 8
    
    ax2.bar3d(dataset[0], dataset[1], bottom, width, depth, dataset[2], shade=True)
    ax2.set_xlabel(r'$n_d$')
    ax2.set_ylabel(r'$n_\theta$')
    ax2.set_title(r'$error$')
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\OLS error {now}"
    plt.savefig(filepath,dpi=300,bbox_inches = 'tight')

def errortest(factor,ntheta,nd,m,errors):
    # function to generate m measurements per error in 'errors',
    # after which we perform OLS to find errors after reconstruction
    u = shepp_logan_phantom()
    uc = compress(u,factor)
    uc_ravel = uc.ravel()
    n = np.shape(u)[0]//factor
    theta = np.linspace(0.0001,np.pi*0.0001,ntheta)
    A = Amatrix(n,theta,nd)
    f_A = A @ uc_ravel
    AA = np.transpose(A) @ A
    radontheta = 180*theta/np.pi
    result = np.zeros((m+1,len(errors)))
    result[0] = errors
    for i in range(0,len(errors)):
        for j in range(0,m):
            f_error = f_A + np.random.normal(loc = 0,scale = errors[i],size = len(f_A))
            u_ols_error = np.linalg.inv(AA) @ np.transpose(A) @ f_error
            u_iradon = iradon(f_error.reshape(nd,ntheta),\
                              radontheta,output_size=n).ravel()
            result[j+1][i] = np.linalg.norm(u_iradon-u_ols_error)/len(f_A)
            
    # saving data
    if not os.path.isdir('Data'):
        os.mkdir("Data")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Data\errortestdata {now}.csv"
    df = pd.DataFrame(result)
    df.to_csv(filepath,header = False, index = False)
    return result,filepath

def errorplot(dataset):
    df = pd.read_csv(dataset)
    header = df.columns.tolist()
    dataset = df.values
    mins = dataset.min(0)
    maxs = dataset.max(0)
    means = dataset.mean(0)
    std = dataset.std(0)
    fig, ax = plt.subplots(1)
    ax.errorbar(header, means, std, fmt='ok', lw=3)
    ax.errorbar(header, means, [means - mins, maxs - means],
                 fmt='.k', ecolor='gray', lw=1)
    
    xmin = -1
    xmax = len(header)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([mins.min()-.1/3,maxs.max()+.1/3])
    ax.set_xlabel(r'Scale of errors added to sinogram')
    ax.set_ylabel(r'$\frac{|| u_{OLS} - u_{iradon}||}{||f||}$')
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\extra error ols {now}"
    plt.savefig(filepath,dpi = 300)
    
def errorexample():
    # function to generate example image of reconstructions of noisy sinograms
    errors = np.linspace(0,.28,8)
    u = shepp_logan_phantom()
    factor = 10
    uc = compress(u,factor)
    uc_ravel = uc.ravel()
    n = np.shape(u)[0]//factor
    ntheta = 2*n
    nd = 2*n
    theta = np.linspace(0.0001,np.pi-0.0001,ntheta)
    A = Amatrix(n,theta,nd)
    f_A = A @ uc_ravel
    AA = np.transpose(A) @ A
    
    # plotting results
    fig, ax = plt.subplots(2,4)
    for i in range(0,len(errors)):
        f_error = f_A + np.random.normal(scale = errors[i],loc=0,size = len(f_A))
        u_ols_error = np.linalg.inv(AA) @ np.transpose(A) @ f_error
        ax_ac = ax[i // 4][i % 4]
        ax_ac.imshow(u_ols_error.reshape(n,n),\
                             extent=(-1,1,-1,1),vmin=0)
        ax_ac.set_xlabel(r'$x$')
        ax_ac.set_ylabel(r'$y$')
        ax_ac.set_title(f"scale = {errors[i]}")
        ax_ac.set_aspect(1)
    fig.tight_layout()
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\error example {now}"
    plt.savefig(filepath,dpi = 300)
              
# code to generate data used in evaluating impact of errors

factor = 20
n = 400 // factor
ntheta = 2*n
nd = 2*n
m = 100
errors = np.linspace(0,.21,8)
result = errortest(factor,ntheta,nd,m,errors)


# Code to generate plots based upon errordata

filepath = 'Data\\errortestdata 2024-06-28 20 09 59.csv'
errorplot(filepath)


# Code to generate plots of noisy reconstructions

errorexample()

def fourtwofour():
    # code to generate example of multicolinearity in section 4.2.4
    n = 2
    theta = np.array([0.00001,np.pi/2])
    nd = n
    A = Amatrix(n, theta, nd,generateimage=True,legend = True)
    print(f"np.linalg.det(np.transpose(A) @ A) = \
          {np.linalg.det(np.transpose(A) @ A)}")
    print(f"A = {A}")
    print(f"np.transpose(A) @ A = {np.transpose(A) @ A}")
    u = np.ones(n*n)
    f = A @ u
    u_ols = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ f
    u_ols = u_ols.reshape(n,n)
    print(f"u_ols = {u_ols}")
    # plot
    fig,ax = plt.subplots(1)
    fig.set_size_inches(4,4)
    
    ax.imshow(u_ols,extent=(-1,1,-1,1),vmin=0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect(1)
    
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\section 424 {now}"
    plt.savefig(filepath,dpi = 300)
    
def fourtwofour_two():
    u = shepp_logan_phantom()
    factor = 10
    unew = compress(u,factor)
    n = np.shape(u)[0]//factor
    ntheta = n*3
    nd = n
    theta = np.linspace(0.001,np.pi,ntheta, endpoint = False)
    A = Amatrix(n,theta,nd)
    
    AA = np.transpose(A) @ A
    print(f"np.linalg.det(AA) = {np.linalg.det(AA)}")
    
    radontheta = 180*theta/np.pi
    f = radon(unew,theta=radontheta)
    
    u_new_raveled = unew.ravel()
    
    f_ols = np.transpose((A @ u_new_raveled).reshape(ntheta,nd))
    
    fnew = f.ravel()
    u_ols = np.linalg.inv(AA) @ np.transpose(A) @ fnew
    
    u_ols_img = u_ols.reshape(n,n)
    
    u_img = iradon(f_ols,theta=radontheta)
    
    fig,axs = plt.subplots(2,2)
    fig.set_size_inches(4,4)
    
    axs[0][0].imshow(f,extent=(0,2*np.pi,-1,1),vmin=0)
    axs[0][0].set_xlabel(r'$\theta$')
    axs[0][0].set_ylabel(r'$s$')
    axs[0][0].set_aspect(np.pi)
    
    axs[0][1].imshow(f_ols,extent=(0,2*np.pi,-1,1),vmin=0)
    axs[0][1].set_xlabel(r'$\theta$')
    axs[0][1].set_ylabel(r'$s$')
    axs[0][1].set_aspect(np.pi)
    
    axs[1][0].imshow(u_ols_img,extent=(-1,1,-1,1),vmin=0)
    axs[1][0].set_xlabel(r'$x$')
    axs[1][0].set_ylabel(r'$y$')
    axs[1][0].set_aspect(1)
    
    axs[1][1].imshow(u_img,extent=(-1,1,-1,1),vmin=0)
    axs[1][1].set_xlabel(r'$x$')
    axs[1][1].set_ylabel(r'$y$')
    axs[1][1].set_aspect(1)
    
    fig.tight_layout()
    
    # saving image
    if not os.path.isdir('Figures'):
        os.mkdir("Figures")
    
    now = datetime.now().strftime('%Y-%m-%d %H %M %S')
    filepath = f"Figures\section 424_2 {now}"
    plt.savefig(filepath,dpi = 300)

fourtwofour()

fourtwofour_two()

# code to generate data used in evaluating forward errors of OLS

factor = 20
n = 400 // factor
ntheta = np.array([n, 3*n//2,2*n,5*n//2,3*n,7*n//2,4*n])
nd = np.array([n,3*n//2,2*n,5*n//2,3*n,7*n//2,4*n])
result = OLS_forward_error_test(factor, ntheta, nd)


# code to generate plot of evaluated forward OLS errors

filepath = 'Data\\ols forward data 2024-06-28 17 22 20.csv'
OLS_forward_error_plot(filepath)


# code to generate data used in evaluating reconstruction errors of OLS

factor = 20
n = 400 // factor
ntheta = np.array([n, 3*n//2,2*n,5*n//2,3*n,7*n//2,4*n])
nd = np.array([n,3*n//2,2*n,5*n//2,3*n,7*n//2,4*n])
result = OLS_reconstruction_error_test(factor, ntheta, nd)


# code to generate plot of evaluated OLS errors

filepath = 'Data\\ols data 2024-06-28 16 31 37.csv'
OLS_reconstruction_error_plot(filepath)