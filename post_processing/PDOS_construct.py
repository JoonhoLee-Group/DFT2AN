#!/usr/bin/env python3
''' Construct the PDOS from outputs of PDOS_compress.py. '''

from funcs import *
from read_output import *
import numpy as np
import sys
import gc
import pickle
import os

folder = sys.argv[1]
scratch_folder = '/n/holylabs/LABS/joonholee_lab/Everyone/lko/'+folder
construct_PDOS_up = True
construct_PDOS_low = False
write_PDOS = False
write_phi = True
compute_Delta_k = False
compute_eta = True
sigma = 0.01
Es = np.arange(-8, 8, 1e-4)
#Es = np.arange(-3, 10, 1e-4)


nk_irr, nmo = get_nk_nmo(folder+'/INPUT.scratch/focka')
eq_kpts = get_eq_kpts(folder+'/OUTPUT.out')
weights = np.array([len(lst) for lst in eq_kpts])
nk = np.sum(weights)

moEa = np.load(scratch_folder+'/moEa.npy')
moEb = np.load(scratch_folder+'/moEb.npy')

##### PDOS_up
if construct_PDOS_up:
    Pa_up_weights = np.load(scratch_folder+'/Pa_up_weights.npy')
    Pb_up_weights = np.load(scratch_folder+'/Pb_up_weights.npy')
    dim = np.shape(Pa_up_weights)[0]
    ###### alpha
    PDOS_up = np.zeros((dim,dim,len(Es)), dtype=complex) 
    if compute_Delta_k:
        Delta_up = np.zeros((dim,dim,len(Es)), dtype=complex)
        H_up = np.zeros((dim,dim), dtype=complex)
        S_up = np.zeros((dim,dim), dtype=complex)
    for k in range(nk_irr):
        gc.collect()
        print(f"Processing k-point {k}")
        # (i-th AO, j-th AO, n-th MO, k_ind, E pt) 
        PDOS_k = np.zeros_like(PDOS_up)
        for i in range(dim):
            for j in range(dim):
                PDOS_k[i,j,:] = np.sum(sigma/np.pi * Pa_up_weights[i,j,:,k,np.newaxis]\
                                        /(np.square(Es[np.newaxis,:]-moEa[:,k,np.newaxis])+sigma**2), axis=0)
        if compute_Delta_k:
            Delta_k, H_k, S_k = weights[k]/nk * phi2DeltaHS(G2phi(PDOS2G(Es, PDOS_k, E_lim=[-200,200])))
            Delta_up += Delta_k
            H_up += H_k
            S_up += S_k
        PDOS_up += weights[k]/nk * PDOS_k

    if write_PDOS:
        np.save(folder+'/PDOS_analysis/PDOSa_up.npy', PDOS_up)
    if compute_Delta_k:
        np.save(folder+'/PDOS_analysis/Deltaa_avg_up.npy', Delta_up)
        np.save(folder+'/PDOS_analysis/Ha_avg_up.npy', H_up)
        np.save(folder+'/PDOS_analysis/Sa_avg_up.npy', S_up)
    if write_phi:
        phi = G2phi(PDOS2G(Es, PDOS_up, E_lim=[-200,200]))
        np.save(scratch_folder+'/phia_up.npy', phi)
    if compute_eta:
        eta = 1/np.pi*argdet(G2phi(PDOS2G(Es, PDOS_up, E_lim=[-200,200])))
        eta = eta - eta[0]
        np.save(scratch_folder+'/etaa_up.npy', eta)
        data = pickle.load(open(folder+'/data.pkl', 'rb'))
        BEa_AN = BE_from_eta(eta, Es, data['mua'])
        data['BEa_AN_up'] = BEa_AN
        pickle.dump(data, open(folder+'/data.pkl', 'wb'))

    ###### beta
    PDOS_up = np.zeros((dim,dim,len(Es)), dtype=complex) 
    if compute_Delta_k:
        Delta_up = np.zeros((dim,dim,len(Es)), dtype=complex)
        H_up = np.zeros((dim,dim), dtype=complex)
        S_up = np.zeros((dim,dim), dtype=complex)
    for k in range(nk_irr):
        gc.collect()
        print(f"Processing k-point {k}")
        # (i-th AO, j-th AO, n-th MO, k_ind, E pt) 
        PDOS_k = np.zeros_like(PDOS_up)
        for i in range(dim):
            for j in range(dim):
                PDOS_k[i,j,:] = np.sum(sigma/np.pi * Pb_up_weights[i,j,:,k,np.newaxis]\
                                        /(np.square(Es[np.newaxis,:]-moEb[:,k,np.newaxis])+sigma**2), axis=0)
        if compute_Delta_k:
            Delta_k, H_k, S_k = weights[k]/nk * phi2DeltaHS(G2phi(PDOS2G(Es, PDOS_k, E_lim=[-200,200])))
            Delta_up += Delta_k
            H_up += H_k
            S_up += S_k
        PDOS_up += weights[k]/nk * PDOS_k

    if write_PDOS:
        np.save(folder+'/PDOS_analysis/PDOSb_up.npy', PDOS_up)
    if compute_Delta_k:
        np.save(folder+'/PDOS_analysis/Deltab_avg_up.npy', Delta_up)
        np.save(folder+'/PDOS_analysis/Hb_avg_up.npy', H_up)
        np.save(folder+'/PDOS_analysis/Sb_avg_up.npy', S_up)
    if write_phi:
        phi = G2phi(PDOS2G(Es, PDOS_up, E_lim=[-200,200]))
        np.save(scratch_folder+'/phib_up.npy', phi)
    if compute_eta:
        eta = 1/np.pi*argdet(G2phi(PDOS2G(Es, PDOS_up, E_lim=[-200,200])))
        eta = eta - eta[0]
        np.save(scratch_folder+'/etab_up.npy', eta)
        data = pickle.load(open(folder+'/data.pkl', 'rb'))
        BEb_AN = BE_from_eta(eta, Es, data['mub'])
        data['BEb_AN_up'] = BEb_AN
        pickle.dump(data, open(folder+'/data.pkl', 'wb'))





##### PDOS_low
if construct_PDOS_low:
    Pa_low_weights = np.load(scratch_folder+'/Pa_low_weights.npy')
    Pb_low_weights = np.load(scratch_folder+'/Pb_low_weights.npy')
    dim = np.shape(Pa_low_weights)[0]
    ###### alpha
    PDOS_low = np.zeros((dim,dim,len(Es)), dtype=complex)
    if compute_Delta_k:
        Delta_low = np.zeros((dim,dim,len(Es)), dtype=complex)
        H_low = np.zeros((dim,dim), dtype=complex)
        S_low = np.zeros((dim,dim), dtype=complex)
    for k in range(nk_irr):
        gc.collect()
        print(f"Processing k-point {k}")
        # (i-th AO, j-th AO, n-th MO, k_ind, E pt) 
        PDOS_k = np.zeros_like(PDOS_low)
        for i in range(dim):
            for j in range(dim):
                PDOS_k[i,j,:] = np.sum(sigma/np.pi * Pa_low_weights[i,j,:,k,np.newaxis]\
                                        /(np.square(Es[np.newaxis,:]-moEa[:,k,np.newaxis])+sigma**2), axis=0)
        if compute_Delta_k:
            Delta_k, H_k, S_k = weights[k]/nk * phi2DeltaHS(G2phi(PDOS2G(Es, PDOS_k, E_lim=[-200,200])))
            Delta_low += Delta_k
            H_low += H_k
            S_low += S_k
        PDOS_low += weights[k]/nk * PDOS_k

    if write_PDOS:
        np.save(folder+'/PDOS_analysis/PDOSa_low.npy', PDOS_low)
    if compute_Delta_k:
        np.save(folder+'/PDOS_analysis/Deltaa_avg_low.npy', Delta_low)
        np.save(folder+'/PDOS_analysis/Ha_avg_low.npy', H_low)
        np.save(folder+'/PDOS_analysis/Sa_avg_low.npy', S_low)
    if write_phi:
        phi = G2phi(PDOS2G(Es, PDOS_low, E_lim=[-200,200]))
        np.save(scratch_folder+'/phia_low.npy', phi)
    if compute_eta:
        eta = 1/np.pi*argdet(G2phi(PDOS2G(Es, PDOS_low, E_lim=[-200,200])))
        eta = eta - eta[0]
        np.save(scratch_folder+'/etaa_low.npy', eta)
        data = pickle.load(open(folder+'/data.pkl', 'rb'))
        BEa_AN = BE_from_eta(eta, Es, data['mua'])
        data['BEa_AN_low'] = BEa_AN
        pickle.dump(data, open(folder+'/data.pkl', 'wb'))

    ###### beta
    PDOS_low = np.zeros((dim,dim,len(Es)), dtype=complex)
    if compute_Delta_k:
        Delta_low = np.zeros((dim,dim,len(Es)), dtype=complex)
        H_low = np.zeros((dim,dim), dtype=complex)
        S_low = np.zeros((dim,dim), dtype=complex)
    for k in range(nk_irr):
        gc.collect()
        print(f"Processing k-point {k}")
        # (i-th AO, j-th AO, n-th MO, k_ind, E pt) 
        PDOS_k = np.zeros_like(PDOS_low)
        for i in range(dim):
            for j in range(dim):
                PDOS_k[i,j,:] = np.sum(sigma/np.pi * Pb_low_weights[i,j,:,k,np.newaxis]\
                                        /(np.square(Es[np.newaxis,:]-moEb[:,k,np.newaxis])+sigma**2), axis=0)
        if compute_Delta_k:
            Delta_k, H_k, S_k = weights[k]/nk * phi2DeltaHS(G2phi(PDOS2G(Es, PDOS_k, E_lim=[-200,200])))
            Delta_low += Delta_k
            H_low += H_k
            S_low += S_k
        PDOS_low += weights[k]/nk * PDOS_k

    if write_PDOS:
        np.save(folder+'/PDOS_analysis/PDOSb_low.npy', PDOS_low)
    if compute_Delta_k:
        np.save(folder+'/PDOS_analysis/Deltab_avg_low.npy', Delta_low)
        np.save(folder+'/PDOS_analysis/Hb_avg_low.npy', H_low)
        np.save(folder+'/PDOS_analysis/Sb_avg_low.npy', S_low)
    if write_phi:
        phi = G2phi(PDOS2G(Es, PDOS_low, E_lim=[-200,200]))
        np.save(scratch_folder+'/phib_low.npy', phi)
    if compute_eta:
        eta = 1/np.pi*argdet(G2phi(PDOS2G(Es, PDOS_low, E_lim=[-200,200])))
        eta = eta - eta[0]
        np.save(scratch_folder+'/etab_low.npy', eta)
        data = pickle.load(open(folder+'/data.pkl', 'rb'))
        BEb_AN = BE_from_eta(eta, Es, data['mub'])
        data['BEb_AN_low'] = BEb_AN
        pickle.dump(data, open(folder+'/data.pkl', 'wb'))











##### DOS_up
# DOSa = np.zeros_like(Es)
# for k in range(nk_irr):
#     print(f"Processing k-point {k}")
#     DOS_k = np.sum(sigma/np.pi * 1/(np.square(Es[np.newaxis,:]-moEa[:,k,np.newaxis])+sigma**2), axis=0)
#     DOSa += weights[k]/nk * DOS_k
# np.save(folder+'/PDOS_analysis/DOSa.npy', DOSa)