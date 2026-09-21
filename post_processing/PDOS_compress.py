#!/usr/bin/env python3
''' write out the mo energies and the PDOS contributions at
those energies. '''

from funcs import *
from read_output import *
import numpy as np
import os
import pickle
import sys


folder = sys.argv[1] 
n_orb_ad = int(sys.argv[2])
if sys.argv[3] == 'True':
    restricted = True
elif sys.argv[3] == 'False':
    restricted = False
else:
    raise ValueError("The third argument must be 'True' or 'False'.")
write_out = True
scratch_folder = '/n/holylabs/LABS/joonholee_lab/Everyone/lko/'+folder

if restricted:
    nk_irr, nmo = get_nk_nmo(folder+'/INPUT.scratch/fock')
else:
    nk_irr, nmo = get_nk_nmo(folder+'/INPUT.scratch/focka')
ad_lst = np.array(range(nmo-n_orb_ad, nmo)) 
eq_kpts = get_eq_kpts(folder+'/OUTPUT.out')
weights = np.array([len(lst) for lst in eq_kpts])
nk = np.sum(weights)
S = np.reshape(read_arma_mat(folder+'/S.arma', dtype=np.complex128), (nmo, nmo, nk_irr), order='F')  # (AO, AO, kpt)

if restricted:
    F = read_arma_cube(folder+'/INPUT.scratch/fock') # (AO, AO, kpt)
    moE, moC = solveEC(F, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
else:
    Fa = read_arma_cube(folder+'/INPUT.scratch/focka') # (AO, AO, kpt)
    Fb = read_arma_cube(folder+'/INPUT.scratch/fockb') # (AO, AO, kpt)
    moEa, moCa = solveEC(Fa, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
    moEb, moCb = solveEC(Fb, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
print('finished solving mo energies')

if write_out:
    try:
        os.makedirs(scratch_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")

if restricted:
    np.save(scratch_folder+'/moEa.npy', moE)
    # contravariant PDOS
    P_up_weights = np.einsum('ink,jnk->ijnk', moC[ad_lst,:,:], moC[ad_lst,:,:].conj())
    np.save(scratch_folder+'/Pa_up_weights.npy', P_up_weights)
    # covariant PDOS
    C_prime = np.einsum('ijk,jnk->ink', S[ad_lst,:,:], moC)
    P_low_weights = np.einsum('ink,jnk->ijnk',C_prime, C_prime.conj())
    np.save(scratch_folder+'/Pa_low_weights.npy', P_low_weights)
else:
    np.save(scratch_folder+'/moEa.npy', moEa)
    np.save(scratch_folder+'/moEb.npy', moEb)
    Pa_up_weights = np.einsum('ink,jnk->ijnk', moCa[ad_lst,:,:], moCa[ad_lst,:,:].conj())
    Pb_up_weights = np.einsum('ink,jnk->ijnk', moCb[ad_lst,:,:], moCb[ad_lst,:,:].conj())
    np.save(scratch_folder+'/Pa_up_weights.npy', Pa_up_weights)
    np.save(scratch_folder+'/Pb_up_weights.npy', Pb_up_weights)
    Ca_prime = np.einsum('ijk,jnk->ink', S[ad_lst,:,:], moCa)
    Cb_prime = np.einsum('ijk,jnk->ink', S[ad_lst,:,:], moCb)
    Pa_low_weights = np.einsum('ink,jnk->ijnk',Ca_prime, Ca_prime.conj())
    Pb_low_weights = np.einsum('ink,jnk->ijnk',Cb_prime, Cb_prime.conj())
    np.save(scratch_folder+'/Pa_low_weights.npy', Pa_low_weights)
    np.save(scratch_folder+'/Pb_low_weights.npy', Pb_low_weights)

