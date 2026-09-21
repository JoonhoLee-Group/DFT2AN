#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:54:57 2026

@author: liwenko
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.signal import hilbert

def dagger(A):
    return np.transpose(np.conjugate(A))

def Gaussian(x, sigma):
    ''' Normalized gaussian function centered at 0 '''
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

def Lorentzian(x, sigma):
    ''' Normalized Lorentzian centered at 0 '''
    # if abs(x) > 50*sigma:
    #     return 0
    return sigma/np.pi * 1/(x**2+sigma**2)
def solve_HF(F, S):
    ''' Solve FC = SCE. 
    Define A = s^(-1/2), and D = S^(1/2)C. Then the equation becomes a standard
    eigenvalue equation
    (AFA)D = DE 
    Returns E (as a vec) and C.
    '''
    A = np.linalg.inv(sqrtm(S))
    AFA = A @ F @ A
    eigvals, eigvecs = np.linalg.eig(AFA)
    sorted_ind = np.argsort(eigvals)
    rtn_E = np.zeros_like(eigvals, dtype=complex)
    rtn_D = np.zeros_like(eigvecs, dtype=complex)
    for i in range(len(sorted_ind)):
        rtn_D[:,i] = eigvecs[:,sorted_ind[i]]
        rtn_E[i] = eigvals[sorted_ind[i]]
    return rtn_E, A@rtn_D 
def Delta_to_Lambda(Epts, Delta_pts, E_lim=(-50, 50)):
    ''' increase E_lim to increase Fourier sampling '''
    if Epts[0] <= E_lim[0] or Epts[-1] >= E_lim[-1]:
        raise Exception("increase E_lim")
    dE = Epts[1] - Epts[0]
    n_low = int((Epts[0]-E_lim[0])//dE)
    n_high = int((E_lim[1]-Epts[-1])//dE)
    extended_Deltas = np.concatenate([np.zeros(n_low), Delta_pts, np.zeros(n_high)])
    extended_Lambdas = np.imag(hilbert(extended_Deltas))
    return extended_Lambdas[n_low:-n_high]


H = np.array([[-0.1, 2.0, 0.5],
              [2.0, 1.2, 0],
              [0.5, 0, 0.4]])
vals, vecs = np.linalg.eig(H)

sigma = 0.01
Es = np.arange(-8, 8, 0.001)
PDOS_L = np.zeros_like(Es)
PDOS_G = np.zeros_like(Es)
for i in range(3):
    Ei = vals[i]
    PDOS_L += np.square(np.abs(vecs[0,i])) * np.array([Lorentzian(E-Ei, sigma) for E in Es])
    PDOS_G += np.square(np.abs(vecs[0,i])) * np.array([Gaussian(E-Ei, sigma) for E in Es])
Deltas_L = np.zeros_like(Es)
Deltas_G = np.zeros_like(Es)
for i in [1,2]:
    Ei = H[i,i]
    Deltas_L += np.pi * np.square(H[0,i]) * np.array([Lorentzian(E-Ei, sigma) for E in Es])
    Deltas_G += np.pi * np.square(H[0,i]) * np.array([Gaussian(E-Ei, sigma) for E in Es])

GI = -np.pi * PDOS_G
GR = -Delta_to_Lambda(Es, GI, E_lim=(-200,200))
phis = 1/(GR + 1j*GI)
Deltas2 = np.imag(phis) - sigma
