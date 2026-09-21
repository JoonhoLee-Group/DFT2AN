#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 15:07:42 2025

@author: liwenko
"""

import numpy as np
from scipy.signal import hilbert, convolve
from scipy.linalg import sqrtm
from scipy.integrate import simpson
from scipy.optimize import fsolve

Ha2eV = 27.211386245988
Bohr2A = 0.5291772105

def dagger(A):
    return np.transpose(np.conjugate(A))

def Gaussian(x, sigma):
    ''' Normalized gaussian function centered at 0 '''
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

def Lorentzian(x, sigma):
    ''' Normalized Lorentzian centered at 0 '''
    # if abs(x) > 50*sigma:
    #     return 0
    return sigma/np.pi * 1/(np.square(x)+np.square(sigma))

def smooth(Es, ys, sigma, method='Lorentzian'):
    dE = Es[1] - Es[0]
    assert sigma > 3*dE
    temp_Es = np.arange(-100*sigma, 100*sigma, dE)
    if method == 'Lorentzian':
        smear_pts = np.array([Lorentzian(E, sigma)*dE for E in temp_Es])
    elif method == 'Gaussian':
        smear_pts = np.array([Gaussian(E, sigma)*dE for E in temp_Es])
    else:
        raise Exception('method can only be Lorentzian or Gaussian')
    return convolve(ys, smear_pts, mode='same')

def Delta_to_Lambda_scalar(Epts, Delta_pts, E_lim=(-50, 50)):
    ''' increase E_lim to increase Fourier sampling '''
    if Epts[0] <= E_lim[0] or Epts[-1] >= E_lim[-1]:
        raise Exception("increase E_lim")
    dE = Epts[1] - Epts[0]
    n_low = int((Epts[0]-E_lim[0])//dE)
    n_high = int((E_lim[1]-Epts[-1])//dE)
    extended_Deltas = np.concatenate([np.zeros(n_low), Delta_pts, np.zeros(n_high)])
    extended_Lambdas = np.imag(hilbert(extended_Deltas))
    return extended_Lambdas[n_low:-n_high]

def Delta_to_Lambda(Epts, Delta_pts, E_lim=(-50,50)):
    if len(np.shape(Delta_pts)) == 1:
        return Delta_to_Lambda_scalar(Epts, Delta_pts, E_lim=E_lim)
    else:
        dim = np.shape(Delta_pts)[0]
        rtn = np.zeros_like(Delta_pts, dtype=complex)
        for i in range(dim):
            for j in range(dim):
                rtn[i,j,:] += Delta_to_Lambda_scalar(Epts, np.real(Delta_pts[i,j,:]), E_lim=E_lim)
                rtn[i,j,:] += 1j*Delta_to_Lambda_scalar(Epts, np.imag(Delta_pts[i,j,:]), E_lim=E_lim)
        return rtn

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

def solveEC(fullF, fullS):
    ''' full F: the full Fock as a cube of shape (nmo, nmo, nk)
        full S: the full overlap as a cube of shape (nmo, nmo, nk)
        Returns E: real-valued matrix of shape (nmo, nk)
                C: complex cube of shape (nao, nmo, nk) '''
    nmo, nk = np.shape(fullF)[1], np.shape(fullF)[2]
    rtnE = np.zeros((nmo, nk), dtype=float)
    rtnC = np.zeros((nmo, nmo, nk), dtype=complex)
    for i in range(nk):
        tempE, tempC = solve_HF(fullF[:,:,i], fullS[:,:,i])
        rtnE[:,i] = np.real(tempE)
        rtnC[:,:,i] = tempC
    return rtnE, rtnC
        
        
def COB(A, B):
    ''' Change matrix A with the change of basis matrix B. 
    Returns A in modified basis. '''
    return dagger(B) @ A @ B

def Lowdin_ortho_COB(S, ind_lst=None):
    ''' Given an overlap matrix S, returns the change of basis (COB) matrix
    that Lowdin orthogonalize the basis. '''
    dim = np.shape(S)[0]
    B = np.identity(dim, dtype=complex)
    ind_lst = np.array(ind_lst)
    if isinstance(ind_lst, np.ndarray):
        S_sub = S[np.ix_(ind_lst, ind_lst)]
        B[np.ix_(ind_lst, ind_lst)] = np.linalg.inv(sqrtm(S_sub))
    else:
        B = np.linalg.inv(sqrtm(S))
    return B

def GS_ortho_COB(S, fixed_ind_lst):
    ''' Given an overlap matrix S, return the COB matrix that fixes the fixed_ind
    , and makes all other basis vectors orthogonal to the fixed_ind basis vectors.
    This function does not assume basis vecs in fixed_ind are orthogonal.
    '''
    dim = np.shape(S)[0]
    B = np.identity(dim, dtype=complex)
    other_ind_lst = np.setdiff1d(np.array(range(dim)), fixed_ind_lst, assume_unique=True)
    inv_Sa = np.linalg.inv(S[np.ix_(fixed_ind_lst, fixed_ind_lst)])
    B[np.ix_(fixed_ind_lst, other_ind_lst)] = -inv_Sa @ S[np.ix_(fixed_ind_lst, other_ind_lst)]
    return B
    
def rel2abs(coord_rel, v1, v2, v3):
    return coord_rel[0]*v1 + coord_rel[1]*v2 + coord_rel[2]*v3

def cell_ind_in_supercell(nk1, nk2, nk3):
    grid1d1 = np.array(range(nk1))
    grid1d2 = np.array(range(nk2))
    grid1d3 = np.array(range(nk3))
    temp = np.zeros((3, len(grid1d1), len(grid1d2), len(grid1d3)), dtype=float)
    for i1 in range(len(grid1d1)):
        for i2 in range(len(grid1d2)):
            for i3 in range(len(grid1d3)):
                temp[:,i1,i2,i3] = [grid1d1[i1], grid1d2[i2], grid1d3[i3]] 
    return np.reshape(temp, (3, -1))

def MP_grid1d(nk, plus_one=False):
    ''' Returns the 1d Monkhorst-Pack grid in relative coordinates given nk points. '''
    assert isinstance(nk, int) and nk > 0
    if nk%2 == 1: # odd
        if plus_one:
            return np.linspace(-(nk//2)/nk, (nk//2 + 1)/nk, nk+1)
        else:
            return np.linspace(-(nk//2)/nk, (nk//2)/nk, nk)
    else:
        if plus_one:
            return np.linspace(-(nk-1)/(2*nk), (nk+1)/(2*nk), nk+1)
        else:
            return np.linspace(-(nk-1)/(2*nk), (nk-1)/(2*nk), nk)
def MP_kpts_rel(nk1, nk2, nk3, flatten=True, plus_one=False):
    ''' Returns a matrix of shape (3, nk1*nk2*nk3). The matrix is a list of
    k points (in relative coord) in qcpbc order. 
    Loop over nk3, then nk2, then nk1.'''
    grid1d1 = MP_grid1d(nk1, plus_one=plus_one)
    grid1d2 = MP_grid1d(nk2, plus_one=plus_one)
    grid1d3 = MP_grid1d(nk3, plus_one=plus_one)
    temp = np.zeros((3, len(grid1d1), len(grid1d2), len(grid1d3)), dtype=float)
    for i1 in range(len(grid1d1)):
        for i2 in range(len(grid1d2)):
            for i3 in range(len(grid1d3)):
                temp[:,i1,i2,i3] = [grid1d1[i1], grid1d2[i2], grid1d3[i3]] 
    if flatten:
        return np.reshape(temp, (3, -1))
    else:
        return temp
def MP_kpts(nk1, nk2, nk3, b1, b2, b3, flatten=True, plus_one=False):
    ''' Returns a list of kpoints as a matrix of shape (3, nk1*nk2*nk3).
    k points in cartesian coordinates, in qcpbc order. '''
    if flatten:
        kpts_rel = MP_kpts_rel(nk1, nk2, nk3, flatten=True, plus_one=plus_one)
        rtn = np.zeros_like(kpts_rel)
        for i in range(np.shape(kpts_rel)[1]):
            rtn[:,i] = rel2abs(kpts_rel[:,i], b1, b2, b3)
        return rtn
    else:
        kpts_rel = MP_kpts_rel(nk1, nk2, nk3, flatten=False, plus_one=plus_one)
        rtn = np.zeros_like(kpts_rel)
        for i1 in range(np.shape(rtn)[1]):
            for i2 in range(np.shape(rtn)[2]):
                for i3 in range(np.shape(rtn)[3]):
                    rtn[:,i1,i2,i3] = rel2abs(kpts_rel[:,i1,i2,i3], b1, b2, b3)
        return rtn

def num_red_kpts(eq_kpts):
    s = 0
    for v in eq_kpts:
        s += len(v)
    return s

def BZ_int_Riemann(f, Epts, eq_kpts, *args):
    ''' Use Riemann sum to integrate f(k, E, *args) over the Brillouin zone.
    Assume kpts are regularly spaced within the BZ. 
    Turns f(k, E, *args) into a function of E, h(E).
    Returns an object of shape (..., len(Epts)).
    f: a function that takes the arguments (k_index, Epts, *args) and outputs
    an array with shape (..., len(Epts))
    Epts: list of E points. An 1D array
    eq_kpts: equivalent k points from symmetry
    '''
    # temp = np.array(f(0, Epts[0], *args))
    # rtn = np.zeros(np.shape(temp) + (len(Epts),), dtype=temp.dtype)
    temp = np.array(f(0, Epts, *args))
    rtn = np.zeros_like(temp)
    nk_red = num_red_kpts(eq_kpts)
    for i in range(len(eq_kpts)):
        print(i)
        rtn += f(i, Epts, *args) * len(eq_kpts[i])
        # for j in range(len(Epts)):
        #     E = Epts[j]
        #     rtn[...,j] += f(i, E, *args)
    return rtn/nk_red

# def PDOS_from_sym(eq_kpts, Epts, mo_E, mo_C, S_full, sigma, ad_ind_lst, smearing='Lorentzian')

# def Lorentzian_DOS_kpt(k_ind, E, mo_E, sigma):
#     eigvals = mo_E[:,k_ind]
#     rtn = 0
#     for eigval in eigvals:
#         rtn += Lorentzian(E-eigval, sigma)
#     return rtn

def DOS_kpt_smear(k_ind, Epts, mo_E, sigma, smearing='Lorentzian'):
    eigvals = mo_E[:,k_ind]
    rtn = np.zeros(len(Epts), dtype=float)
    for eigval in eigvals:
        if smearing == 'Lorentzian':
            rtn += [Lorentzian(E-eigval, sigma) for E in Epts]
        elif smearing == 'Gaussian':
            rtn += [Gaussian(E-eigval, sigma) for E in Epts]
        else: 
            raise Exception('unrecognized smearing method')
    return rtn

def PDOS_kpt_smear(k_ind, Epts, mo_E, mo_C, S_full, sigma, ad_ind_lst, \
                   smearing='Lorentzian', covariant=True):
    eigvals = mo_E[:,k_ind]
    C = mo_C[:,:,k_ind]
    rtn = np.zeros((len(ad_ind_lst),len(ad_ind_lst),len(Epts)), dtype=complex)
    if covariant:
        S_sub = S_full[:,ad_ind_lst,k_ind]
    for i in range(len(eigvals)):
        eigval = eigvals[i]
        if covariant:
            v = C[:,i]
            temp_dm = dagger(S_sub) @ np.outer(v, np.conjugate(v)) @ S_sub 
        else:
            v = C[ad_ind_lst,i]
            temp_dm = np.outer(v, np.conjugate(v))
        if smearing == 'Lorentzian':
            profile = [Lorentzian(E-eigval, sigma) for E in Epts]
        elif smearing == 'Gaussian':
            profile = [Gaussian(E-eigval, sigma) for E in Epts]
        else:
            raise Exception('unrecognized smearing method')
        for j in range(len(Epts)):
            rtn[:,:,j] += temp_dm * profile[j]
    return rtn

def Delta2PDOS(Epts, Deltas, H_a, S_a, eps, E_lim=(-50, 50)):
    ''' Convert Delta(E) to PDOS(E)
    Assume Delta is in an orthonormal basis.
    If scalar=True, Deltas is an 1D array of the same size as Epts, and H_a is 
    a scalar.
    If scalar=False, Deltas is a 3D array of size (nmo,nmo,len(Epts)), and H_a 
    is a matrix.
    Returns PDOS with the same shape as Deltas
    '''
    # print('set eps=', eps, 'to be equal to sigma for optimal result')
    # print('Delta2PDOS assumes the input Delta is in an orthonormal basis!')
    if len(np.shape(Deltas)) == 1:
        assert np.issubdtype(Deltas.dtype, np.floating)
        Lambdas = Delta_to_Lambda(Epts, Deltas, E_lim=E_lim)
        Gs = 1/((Epts+1j*eps)*S_a-H_a-Lambdas+1j*Deltas)
        return -1/np.pi * np.imag(Gs)
    else:
        Lambdas = np.zeros_like(Deltas, dtype=complex)
        dim = np.shape(Deltas)[0]
        for i in range(dim):
            for j in range(dim):
                Lambdas[i,j,:] += Delta_to_Lambda(Epts, np.real(Deltas[i,j,:]), E_lim=E_lim)
                Lambdas[i,j,:] += 1j*Delta_to_Lambda(Epts, np.imag(Deltas[i,j,:]), E_lim=E_lim)
        rtn = np.zeros_like(Deltas, dtype=complex)
        # phis = np.zeros_like(Deltas, dtype=complex)
        for i in range(len(Epts)):
            phi = (Epts[i]+1j*eps) * S_a
            phi = phi - H_a - Lambdas[:,:,i] + 1j*Deltas[:,:,i]
            # phis[:,:,i] = phi
            G = np.linalg.inv(phi)
            rtn[:,:,i] = 1j/(2*np.pi) * (G-dagger(G))
        return rtn

def PDOS2G(Epts, PDOS, E_lim=(-50,50)):
    ''' Returns G '''
    if len(np.shape(PDOS)) == 1:
        B = -np.pi*PDOS
        A = -Delta_to_Lambda(Epts, np.real(B), E_lim=E_lim) \
            -1j*Delta_to_Lambda(Epts, np.imag(B), E_lim=E_lim)
        return A+1j*B
    else:
        dim = np.shape(PDOS)[0]
        B = -np.pi*PDOS
        A = np.zeros_like(B)
        for i in range(dim):
            for j in range(dim):
                A[i,j,:] = -Delta_to_Lambda(Epts, np.real(B[i,j,:]), E_lim=E_lim) \
                          -1j*Delta_to_Lambda(Epts, np.imag(B[i,j,:]), E_lim=E_lim)
        return A + 1j*B
    #     B_R = -np.pi * np.real(PDOS)
    #     B_I = -np.pi * np.imag(PDOS)
    #     A_R = np.zeros_like(B_R)
    #     A_I = np.zeros_like(B_I)
    #     for i in range(dim):
    #         for j in range(dim):
    #             A_R[i,j,:] = -Delta_to_Lambda(Epts, B_R[i,j,:], E_lim=E_lim)
    #             A_I[i,j,:] = -Delta_to_Lambda(Epts, B_I[i,j,:], E_lim=E_lim)
    #     G = A_R+1j*A_I-B_I+1j*B_R
    # return G

def G2PDOS(Es, G):
    if len(np.shape(G)) == 1:
        return -1/np.pi * np.imag(G)
    else:
        rtn = np.zeros_like(G, dtype=complex)
        for i in range(len(Es)):
            temp = G[:,:,i]
            rtn[:,:,i] = -1/np.pi * (temp-dagger(temp))/(2j)
        return rtn

def G2phi(Gs):
    if len(np.shape(Gs)) == 1:
        return 1/Gs
    else:
        phis = np.moveaxis(np.linalg.inv(np.moveaxis(Gs,-1,0)),0,-1)
        # phis = np.zeros_like(Gs)
        # for i in range(np.shape(Gs)[2]):
        #     phis[:,:,i] = np.linalg.inv(Gs[:,:,i])
        return phis

def argdet(phis):
    if len(np.shape(phis)) == 1:
        return np.unwrap(np.angle(phis))
    else:
        detphis = np.linalg.det(np.moveaxis(phis, -1, 0))
        # np.zeros(np.shape(phis)[2], dtype=complex)
        # for i in range(np.shape(phis)[2]):
        #     detphis[i] = np.linalg.det(phis[:,:,i])
        return np.unwrap(np.angle(detphis))

def phi2eta(phis):
    rtn = 1/np.pi * argdet(phis)
    rtn = rtn - rtn[0]
    return rtn

def phi2Delta(phis):
    if len(np.shape(phis)) == 1:
        return np.imag(phis)
    else:
        Deltas = np.zeros_like(phis)
        for i in range(np.shape(phis)[2]):
            Deltas[:,:,i] = (phis[:,:,i]-dagger(phis[:,:,i]))/(2j)
        Deltas = Deltas-Deltas[:,:,[100]]
        return Deltas

def phi2DeltaHS(phis, Es, E_lim=(-50,50), offset_E_ind=100):
    ''' Returns H and S '''
    Deltas = phi2Delta(phis)
    if len(np.shape(phis)) == 1:
        Deltas = Deltas - Deltas[offset_E_ind]
    else:
        Deltas = Deltas - Deltas[:,:,offset_E_ind,np.newaxis]
    ind1, ind2 = len(Es)//4, len(Es)//4*3
    if len(np.shape(phis)) == 1:
        Lambdas = Delta_to_Lambda_scalar(Es, Deltas, E_lim=E_lim)
        temp = np.real(phis) + Lambdas
        m, b = np.polyfit(Es[ind1:ind2], temp[ind1:ind2], 1)
        return Deltas, -b, m
    else:
        Lambdas = Delta_to_Lambda(Es, Deltas, E_lim=E_lim)
        temp = np.zeros_like(phis, dtype=complex)
        for i in range(len(Es)):
            temp[:,:,i] = (phis[:,:,i] + dagger(phis[:,:,i]))/2 + Lambdas[:,:,i]
        dim = np.shape(phis)[0]
        Heff = np.zeros((dim,dim), dtype=complex)
        Seff = np.zeros((dim,dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                m, b = np.polyfit(Es[ind1:ind2], temp[i,j,ind1:ind2], 1)
                Heff[i,j] = -b
                Seff[i,j] = m
        return Deltas, Heff, Seff
            
def PDOS2Delta(Epts, PDOS, eps, E_lim=(-50,50)):
    ''' Returns H_a and Deltas '''
    print('PDOS2Delta assumes the input PDOS is in an orthonormal basis!')
    print('PDOS2Delta assumes the input PDOS is normalized!')
    if len(np.shape(PDOS)) == 1:
        temp = simpson(PDOS, x=Epts)
        print('scalar PDOS integrates to', round(temp, 5))
        B = -np.pi*PDOS
        A = -Delta_to_Lambda(Epts, B, E_lim=E_lim)
        phi = 1/(A+1j*B)
        Deltas = np.imag(phi) - eps
        Lambdas = Delta_to_Lambda(Epts, Deltas, E_lim=E_lim)
        Ea_lst = Epts - Lambdas - np.real(phi)
        return np.average(Ea_lst), Deltas
    else:
        dim = np.shape(PDOS)[0]
        temp = np.trace(PDOS) / dim
        temp2 = simpson(np.real(temp), x=Epts)
        print('Average PDOS integral is', round(temp2, 5))
        B_R = -np.pi * np.real(PDOS)
        B_I = -np.pi * np.imag(PDOS)
        A_R = np.zeros_like(B_R)
        A_I = np.zeros_like(B_I)
        for i in range(dim):
            for j in range(dim):
                A_R[i,j,:] = -Delta_to_Lambda(Epts, B_R[i,j,:], E_lim=E_lim)
                A_I[i,j,:] = -Delta_to_Lambda(Epts, B_I[i,j,:], E_lim=E_lim)
        G = A_R+1j*A_I-B_I+1j*B_R
        
        phis = np.zeros((dim,dim,len(Epts)), dtype=complex)
        Deltas = np.zeros((dim,dim,len(Epts)), dtype=complex)
        for i in range(len(Epts)):
            phi = np.linalg.inv(G[:,:,i])
            Delta = (phi-dagger(phi))/(2j) - eps*np.identity(dim)
            phis[:,:,i] = phi
            Deltas[:,:,i] = Delta
        
        Lambdas = np.zeros((dim,dim,len(Epts)), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                Lambdas[i,j,:] = Delta_to_Lambda(Epts, np.real(Deltas[i,j,:]), E_lim=E_lim)
                Lambdas[i,j,:] += 1j*Delta_to_Lambda(Epts, np.imag(Deltas[i,j,:]), E_lim=E_lim)
                
        H_a_lst = np.zeros((dim,dim,len(Epts)), dtype=complex)
        for i in range(len(Epts)):
            phi = phis[:,:,i]
            temp = (phi+dagger(phi)) / 2
            H_a_lst[:,:,i] = Epts[i]*np.identity(dim, dtype=complex) - Lambdas[:,:,i] - temp 
        
        return np.average(H_a_lst[:,:,len(Epts)//4:len(Epts)//4*3], axis=2), Deltas

def BE_from_eta(eta, Es, mu):
    mu_ind = np.argmin(np.abs(Es-mu))
    return simpson(eta[:mu_ind+1], x=Es[:mu_ind+1])

def binding_E(argdetphis, Es, mu, GC=True):
    assert mu > Es[0] and mu < Es[-1]
    mu_ind = np.argmin(np.abs(Es-mu))
    if GC:
        temp = (Es[0]-mu) * argdetphis[0]
        temp += simpson(argdetphis[:mu_ind+1], x=Es[:mu_ind+1])
        return temp / np.pi
    else:
        temp = Es[0]*argdetphis[0] - mu*argdetphis[mu_ind]
        temp += simpson(argdetphis[:mu_ind+1], x=Es[:mu_ind+1])
        return temp / np.pi

def set_zero(A, block_degen, block_dim, zero_value=0):
    mask = np.zeros_like(A, dtype=bool)
    ind1 = 0
    for i in range(len(block_degen)):
        for _ in range(block_degen[i]):
            ind2 = ind1 + block_dim[i]
            mask[ind1:ind2, ind1:ind2] = True
            ind1 = ind2
    A[~mask] = zero_value
    
def average_blocks(A, block_degen, block_dim):
    ind1 = 0
    for i in range(len(block_degen)):
        d = block_dim[i]
        temp = np.zeros((d, d, block_degen[i]), dtype=A.dtype)
        for j in range(block_degen[i]):
            temp[:,:,j] = A[ind1+j*d:ind1+(j+1)*d, ind1+j*d:ind1+(j+1)*d]
        avg_temp = np.average(temp, axis=2)
        for j in range(block_degen[i]):
            A[ind1+j*d:ind1+(j+1)*d, ind1+j*d:ind1+(j+1)*d] = avg_temp
        ind1 += block_degen[i] * d

def S_loc_from_sym(S_full, ad_ind_lst, eq_kpts, symmetry_COB, \
                   block_degen, block_dim, sym_basis=True, set_real=True):
    ''' Returns the local overlap matrix from symmetry calculations.
    symmetry_COB: the change of basis matrix V 
                (A in qcpbc basis --> V^dag A V in symmetry adapted basis)
    block_degen: a 1d array of degeneracies of diagonal blocks
    block_dim: a 1d array of diagonal block dimensions
    If sys_basis=True, the returned S_loc is in the symmetry adapted basis. 
    Otherwise, the returned S_loc is in the qcpbc basis. '''
    nk = 0
    for k_class in eq_kpts:
        nk += len(k_class)
    dim = len(ad_ind_lst)
    S_loc = np.zeros((dim, dim), dtype=complex)
    for i in range(len(eq_kpts)):
        S_loc += len(eq_kpts[i]) * (S_full[:,:,i])[np.ix_(ad_ind_lst,ad_ind_lst)]
    S_loc /= nk
    S_loc = COB(S_loc, symmetry_COB)
    set_zero(S_loc, block_degen, block_dim)
    average_blocks(S_loc, block_degen, block_dim)
    if set_real:
        print('setting S_loc to be real by conjugate symmetry')
        S_loc = np.real(S_loc).astype(np.complex128)
    if sym_basis:
        return S_loc
    else:
        return COB(S_loc, dagger(symmetry_COB))

def PDOS_from_sym(Epts, mo_E, mo_C, S_full, sigma, ad_ind_lst, \
                  eq_kpts, symmetry_COB, block_degen, block_dim, \
                  smearing='Lorentzian', sym_basis=True, set_real=True, covariant=True):
    ''' Returns the PDOS from symmetry calculations. '''
    nk = 0
    for k_class in eq_kpts:
        nk += len(k_class)
    dim = len(ad_ind_lst)
    rtn = np.zeros((dim,dim,len(Epts)), dtype=complex)
    for i in range(len(eq_kpts)):
        degen = len(eq_kpts[i])
        rtn += degen * PDOS_kpt_smear(i, Epts, mo_E, mo_C, S_full, sigma, \
                                      ad_ind_lst, smearing=smearing, covariant=covariant) / nk
    for i in range(len(Epts)):
        temp = COB(rtn[:,:,i], symmetry_COB)
        set_zero(temp, block_degen, block_dim)
        average_blocks(temp, block_degen, block_dim)
        rtn[:,:,i] = temp
    if set_real:
        print('setting PDOS to be real by conjugate symmetry')
        rtn = np.real(rtn).astype(np.complex128)
    if sym_basis:
        return rtn
    else:
        for i in range(len(Epts)):
            rtn[:,:,i] = COB(rtn[:,:,i], dagger(symmetry_COB))
        return rtn

def occ_num_single(E, mu, T):
    temp = (E-mu)/T
    if temp > 25:
        return 0
    elif temp < -25:
        return 1
    else:
        return 1/(np.exp(temp)+1)
    
def occ_num(mo_E, mu, T):
    nmo, nk = np.shape(mo_E)[0], np.shape(mo_E)[1]
    rtn = np.zeros_like(mo_E)
    for i in range(nmo):
        for j in range(nk):
            rtn[i,j] = occ_num_single(mo_E[i,j], mu, T)
    return rtn

def density_mat(C_full, occ_nums):
    nao, nk = np.shape(C_full)[0], np.shape(C_full)[2]
    rtn = np.zeros((nao, nao, nk), dtype=complex)
    for i in range(nk):
        rtn[:,:,i] = C_full[:,:,i] @ np.diag(occ_nums[:,i]) @ dagger(C_full[:,:,i])
    return rtn

def get_N(mu, T, mo_E, weights):
    return np.average(np.sum(occ_num(mo_E, mu, T), axis=(0)), weights=weights)

def get_N_a_plus_b(mu, T, mo_Ea, mo_Eb, weights):
    return get_N(mu, T, mo_Ea, weights) + get_N(mu, T, mo_Eb, weights)

def get_mu(N, mo_E, T, weights, mu_guess):
    def helper(mu, T, mo_E, weights, N):
        return get_N(mu, T, mo_E, weights) - N
    return fsolve(helper, mu_guess, args=(T, mo_E, weights, N))

def get_mu_a_plus_b(N, mo_Ea, mo_Eb, T, weights, mu_guess):
    def helper(mu, T, mo_Ea, mo_Eb, weights, N):
        return get_N_a_plus_b(mu, T, mo_Ea, mo_Eb, weights) - N
    return fsolve(helper, mu_guess, args=(T, mo_Ea, mo_Eb, weights, N))


def Fourier_E2T(Es, fs, E_lim=(-50,50)):
    ''' f is a function of E '''
    dE = Es[1] - Es[0]
    n_low = int((Es[0]-E_lim[0])//dE)
    n_high = int((E_lim[1]-Es[-1])//dE)
    extended_Es = np.arange(Es[0]-n_low*dE, Es[-1]+(n_high+0.5)*dE, dE)
    extended_fs = np.concatenate([np.zeros(n_low), fs, np.zeros(n_high)])
    ts = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(extended_Es), dE)) 
    f_FT = dE * (np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.real(extended_fs)))) \
                 + 1j*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.imag(extended_fs)))))
    f_FT = f_FT * np.exp(-1j*np.fft.ifftshift(extended_Es)[0]*ts)
    return ts, f_FT
    



# test
# import matplotlib.pyplot as plt

# Es = np.arange(-50, 50, 0.01)
# fs = np.array([Lorentzian(E-3, 2) for E in Es])
# ts, fs_T = Fourier_E2T(Es, fs, E_lim=[-200, 200])
# plt.plot(Es, fs)
# plt.plot(ts, np.real(fs_T))
# plt.plot(ts, np.exp(-2*ts)*np.heaviside(ts, 0)*np.cos(3*ts))

# Es = np.load('test_Es.npy')
# Deltas = np.load('test_Deltas.npy')
# PDOS = np.load('test_PDOS.npy')
# H_a = np.load('test_H_a.npy')

# Es = np.load('test2_Es.npy')
# Deltas = np.load('test2_Deltas.npy')
# PDOS = np.load('test2_PDOS.npy')
# H_a = np.load('test2_H_a.npy')
# sigma = 0.01

# # PDOS_c = Delta2PDOS(Es, Deltas, H_a, sigma)
# H_a_c, Deltas_c = PDOS2Delta(Es, PDOS, sigma, E_lim=(-50,50))

# # i,j = 0,1
# # plt.plot(Es, np.real(PDOS_c[i,j,:]))
# # plt.plot(Es, np.real(PDOS[i,j,:]))
# # plt.plot(Es, np.imag(PDOS_c[i,j,:]))
# # plt.plot(Es, np.imag(PDOS[i,j,:]))
# # i,j = 1,0
# # plt.plot(Es, np.real(Deltas_c[i,j,:]))
# # plt.plot(Es, np.real(Deltas[i,j,:]))
# # plt.plot(Es, np.imag(Deltas_c[i,j,:]))
# # plt.plot(Es, np.imag(Deltas[i,j,:]))
# # plt.figure()
# # plt.imshow(np.imag(H_a))
# plt.figure()
# plt.imshow(np.abs(H_a_c-H_a))
# plt.figure()
# plt.imshow(np.abs(H_a))
# print(np.sum(np.abs(H_a-H_a_c)))

# diff = np.zeros_like(H_a, dtype=float)
# for i in range(26):
#     for j in range(26):
#         if np.abs(H_a[i,j]) > 1e-6:
#             diff[i,j] = np.abs(H_a_c[i,j]-H_a[i,j]) / np.abs(H_a[i,j])
# plt.imshow(diff)

# check scalar Delta to PDOS
# sigma = 0.1
# V = 0.8
# Es = np.arange(-5, 5, 0.001)
# H = np.array([[0,V],[V,1]])
# vals, vecs = np.linalg.eig(H)
# PDOS_L = np.square(np.abs(vecs[0,0])) * np.array([Lorentzian(E-vals[0], sigma) for E in Es])
# PDOS_L += np.square(np.abs(vecs[0,1])) * np.array([Lorentzian(E-vals[1], sigma) for E in Es])
# PDOS_G = np.square(np.abs(vecs[0,0])) * np.array([Gaussian(E-vals[0], sigma) for E in Es])
# PDOS_G += np.square(np.abs(vecs[0,1])) * np.array([Gaussian(E-vals[1], sigma) for E in Es])
# Ds_L = V**2*np.pi*np.array([Lorentzian(E-1, sigma) for E in Es])
# Ds_G = V**2*np.pi*np.array([Gaussian(E-1, sigma) for E in Es])
# Lambda_L = Delta_to_Lambda(Es, Ds_L)
# Lambda_G = Delta_to_Lambda(Es, Ds_G)
# # PDOS_L = Delta2PDOS(Es, Ds_L, 0, scalar=True, eps=sigma, E_lim=(-200,200))
# # PDOS_G = Delta2PDOS(Es, Ds_G, 0, scalar=True, eps=sigma, E_lim=(-100,100))
# Ea_L, Ds_Lc = PDOS2Delta(Es, PDOS_L, sigma, scalar=True)
# Ea_G, Ds_Gc = PDOS2Delta(Es, PDOS_G, sigma, scalar=True)
# plt.plot(Es, Ds_Gc)
# plt.plot(Es, Ds_G, linestyle='dashed')
# # plt.plot(Es, PDOS_L)
# # plt.plot(Es, PDOS_G)
# # plt.plot(Es, PDOS, linestyle='dashed')
# plt.xlabel('E')
# plt.ylabel(r'$\Delta(E)$')
# plt.legend(['Gaussian smeared PDOS --> Delta', 'Gaussian smeared Delta'])

# G1 = np.zeros((2,2,len(Es)), dtype=complex)
# G2 = np.zeros_like(G1)
# for i in range(len(Es)):
#     temp = Es[i]*np.identity(2, dtype=complex) - H
#     temp1 = temp + 1j*np.array([[sigma,0],[0,2*sigma]])
#     temp2 = temp + 1j*np.array([[sigma,0],[0,sigma]])
#     G1[:,:,i] = np.linalg.inv(temp1)
#     G2[:,:,i] = np.linalg.inv(temp2)
# plt.plot(Es, -1/np.pi*np.imag(G1[0,0,:]))
# plt.plot(Es, -1/np.pi*np.imag(G2[0,0,:]))

# atoms2-band flat eps coupling to mimic eps term
# Es = np.arange(-30, 30, 0.001)
# D2 = sigma*np.ones(len(Es), dtype=float)
# PDOS_2 = Delta2PDOS(Es, D2, 1, 0, scalar=True)
# # plt.plot(Es, PDOS_2)
# D1_scalar = np.pi*V**2*PDOS_2
# PDOS_1_scalar = Delta2PDOS(Es, D1_scalar, 0, sigma, scalar=True)
# plt.plot(Es, PDOS_1_scalar)

# D1_mat = np.zeros((2,2,len(Es)), dtype=complex)
# D1_mat[1,1,:] = 2*D2
# D1_mat[0,0,:] = D2
# PDOS_1_mat = Delta2PDOS(Es, D1_mat, H, 0)
# plt.plot(Es, np.real(PDOS_1_mat[0,0,:]))



# check matrix Delta to PDOS
# PDOS_c = Delta2PDOS(Es, Deltas, F4[:26,:26])
# plt.plot(Es, PDOS_c[0,3,:])
# plt.plot(Es, PDOS[0,3,:])

