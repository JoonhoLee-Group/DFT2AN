#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:03:57 2025

@author: liwenko
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, ceil, e
from scipy.integrate import simpson, cumulative_simpson
from scipy.optimize import root_scalar
import scipy.stats as stats
from scipy.signal import hilbert


def Delta(E, V, alpha, beta):
    if E <= alpha-2*beta or E >= alpha+2*beta:
        return 0
    else:
        return V**2/beta * np.sqrt(1-np.square((E-alpha)/(2*beta)))
    
def Lambda(E, V, alpha, beta):
    a = (E-alpha)/(2*beta)
    if a >= -1 and a <= 1:
        return V**2 * a / beta
    elif a > 1:
        return V**2/beta * (a-np.sqrt(a**2-1))
    else:
        return V**2/beta * (a+np.sqrt(a**2-1))

def phi(E, Ea, V, alpha, beta, eps=1e-4):
    return E-Ea-Lambda(E, V, alpha, beta) +1j*(eps+Delta(E, V, alpha, beta))

def check_lower_isolated(Ea, V, alpha, beta):
    ''' Returns whether there is a isolated lower eigenvalue in H '''
    E_LE = alpha - 2*beta
    return Ea < E_LE + V**2/beta

def check_higher_isolated(Ea, V, alpha, beta):
    ''' Returns whether there is a isolated higher eigenvalue in H '''
    E_UE = alpha + 2*beta
    return Ea > E_UE - V**2/beta

def DeltaN(mu, Ea, V, alpha, beta, eps=1e-4):
    return -np.angle(phi(mu, Ea, V, alpha, beta, eps=eps))/pi

def DeltaE(mu, Ea, V, alpha, beta, eps=1e-4, dE=1e-2):
    lower_isolated = check_lower_isolated(Ea, V, alpha, beta)
    higher_isolated = check_higher_isolated(Ea, V, alpha, beta)
    if lower_isolated:
        E_minus = solve_lower_root(Ea, V, alpha, beta)
    if higher_isolated:
        E_plus = solve_upper_root(Ea, V, alpha, beta)
    # mu below band edge cases
    E_LE = alpha - 2*beta # lower band edge
    E_UE = alpha + 2*beta # upper band edge
    if mu < E_LE:
        if not lower_isolated:
            return -Ea
        else:
            return -Ea + E_minus*(mu >= E_minus)
    # mu above band edge
    # E_L
    if lower_isolated:
        E_L = E_minus
    else:
        E_L = E_LE
    # integral (divided by pi)
    integral = 0
    E1 = E_LE
    if mu >= E_LE and mu < E_UE:
        E2 = mu
    else:
        E2 = E_UE
        if higher_isolated:
            integral += min(mu, E_plus) - E_UE
    Es = np.arange(E1, E2+dE/2, dE)
    phis = np.array([phi(E, Ea, V, eps=eps) for E in Es])
    integral += simpson(np.angle(phis), x=Es)/pi
    return E_L - Ea + integral - mu*np.angle(phi(mu, Ea, V, eps=eps))/pi

def DeltaE2(mu, Ea, V, alpha, beta, eps=1e-4, dE=1e-2):
    E_L = min(Ea - V**2/beta, alpha-2*beta, mu) - 1
    Es = np.arange(E_L, mu+dE/2, dE)
    phis = np.array([phi(E, Ea, V, eps=eps) for E in Es])
    integral = simpson(np.angle(phis), x=Es)
    return E_L + (-mu*np.angle(phis[-1]) + integral)/pi - Ea

def DeltaFE(mu, Ea, V, alpha, beta, eps=1e-4, dE=1e-2):
    ''' Returns the adsorption free energy. '''
    return DeltaE(mu, Ea, V, alpha, beta, eps=eps, dE=dE) - mu*DeltaN(mu, Ea, V, alpha, beta, eps=eps)

def solve_upper_root(Ea, V, alpha, beta, offset=0.1):
    E_UE = alpha + 2*beta # upper band edge
    if check_higher_isolated(Ea, V, alpha, beta):
        def tempfunc(E):
            return E - Ea - Lambda(E, V, alpha, beta)
        return root_scalar(tempfunc, bracket=[E_UE, Ea+V**2/beta+0.1]).root
    else:
        return None

def solve_lower_root(Ea, V, alpha, beta, offset=0.1):
    E_LE = alpha - 2*beta # lower band edge
    if check_lower_isolated(Ea, V, alpha, beta):
        def tempfunc(E):
            return E - Ea - Lambda(E, V, alpha, beta)
        return root_scalar(tempfunc, bracket=[Ea-V**2/beta-offset, E_LE]).root
    else:
        return None

def PDOS(E, Ea, V, alpha, beta, eps=1e-4):
    ''' Returns the adsorbate projected density of states. '''
    phi_temp = E-Ea-Lambda(E, V, alpha, beta) + 1j*(eps+Delta(E, V, alpha, beta))
    return - np.imag(1/phi_temp) / pi

def popa(Ea, V, mu, alpha, beta, eps=1e-3, dE=1e-4):
    ''' Returns the adsorbate population. '''
    E_LE = alpha - 2*beta
    if check_lower_isolated(Ea, V, alpha, beta):
        E1 = solve_lower_root(Ea, V, alpha, beta) - 30*eps
    else:
        E1 = E_LE - 30*eps
    Es = np.arange(E1, mu+dE/2, dE)
    PDOSs = np.array([PDOS(E, Ea, V, alpha, beta, eps=eps) for E in Es])
    return simpson(PDOSs, x=Es)


def bindingE_arg(Es, Ea, V, mu, alpha, beta, eps=1e-4):
    phis = np.array([phi(E, Ea, V, alpha, beta, eps=eps) for E in Es])
    argphis = np.unwrap(np.angle(phis))
    E_ind_mu = np.argmin(np.abs(Es-mu))
    temp1 = simpson(argphis[:E_ind_mu+1]-argphis[0], x=Es[:E_ind_mu+1])
    # temp2 = Es[0]*argphis[0]
    return temp1/pi - Ea

def bindingE_arctan(Es, Ea, V, mu, alpha, beta):
    Deltas = np.array([Delta(E, V, alpha, beta) for E in Es])
    Lambdas = np.array([Lambda(E, V, alpha, beta) for E in Es])
    # temp = np.arctan((Deltas+1e-4)/(Es-Ea-Lambdas)) / pi
    temp = (np.arctan2(Deltas+1e-4, Es-Ea-Lambdas)-pi) / pi
    E_ind_mu = np.argmin(np.abs(Es-mu))
    # E_ind_LE = np.argmin(np.abs(Es+2))
    temp1 = simpson(temp[:E_ind_mu+1], x=Es[:E_ind_mu+1])
    return temp1 - Ea

    
def bindingE_ad_PDOS(Es, Ea, V, mu, alpha, beta):
    PDOSs = np.array([PDOS(E, Ea, V, alpha, beta, eps=0.01) for E in Es])
    E_ind_mu = np.argmin(np.abs(Es-mu))
    E_PDOS = Es*PDOSs
    temp = simpson(E_PDOS[:E_ind_mu+1], x=Es[:E_ind_mu+1])
    return temp - Ea

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

