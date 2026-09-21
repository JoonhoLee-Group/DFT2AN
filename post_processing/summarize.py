#!/usr/bin/env python3

from funcs import *
from read_output import *
import numpy as np
import sys 
import pickle
import os

folder = sys.argv[1]
if sys.argv[2] == 'True':
    restricted = True
elif sys.argv[2] == 'False':
    restricted = False
else:
    raise ValueError("The second argument must be 'True' or 'False'.")
T = 0.001
adjust_mu = False

Na, Nb = get_N(folder+'/OUTPUT.out')
mus = get_EFermi(folder+'/OUTPUT.out')
if isinstance(mus, tuple):
    separate_mu = True
    mua, mub = mus
else:
    separate_mu = False
    mua = mus
    mub = mus
E_total = get_totalE(folder+'/OUTPUT.out')
if restricted:
    nk_irr, nmo = get_nk_nmo(folder+'/INPUT.scratch/fock')
else:
    nk_irr, nmo = get_nk_nmo(folder+'/INPUT.scratch/focka')
eq_kpts = get_eq_kpts(folder+'/OUTPUT.out')
weights = np.array([len(lst) for lst in eq_kpts])
S = np.reshape(read_arma_mat(folder+'/S.arma', dtype=np.complex128), (nmo, nmo, nk_irr), order='F')  # (AO, AO, kpt)
if restricted:
    F = read_arma_cube(folder+'/INPUT.scratch/fock') # (AO, AO, kpt)
    moE, moC = solveEC(F, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
    occ = occ_num(moE, mua, T)
    dm = np.einsum('ijk,jk,ljk->ilk', moC, occ, moC.conj()) # (AO, AO, kpt)
    E_eig = np.real(np.average(np.einsum('ijm,jim->m', F, dm), weights=weights))
else:
    Fa = read_arma_cube(folder+'/INPUT.scratch/focka') # (AO, AO, kpt)
    Fb = read_arma_cube(folder+'/INPUT.scratch/fockb') # (AO, AO, kpt)
    moEa, moCa = solveEC(Fa, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
    moEb, moCb = solveEC(Fb, S) # mo_E:(MO, kpt). mo_C:(AO, MO, kpt)
    occa = occ_num(moEa, mua, T)
    occb = occ_num(moEb, mub, T)
    dma = np.einsum('ijk,jk,ljk->ilk', moCa, occa, moCa.conj()) # (AO, AO, kpt)
    dmb = np.einsum('ijk,jk,ljk->ilk', moCb, occb, moCb.conj()) # (AO, AO, kpt)
    E_eig = np.real(np.average(np.einsum('ijm,jim->m', Fa, dma) + np.einsum('ijm,jim->m', Fb, dmb), weights=weights))


temp = np.loadtxt(folder+'/INPUT.scratch/energies', usecols=[-1])[1:]
if len(temp) == 7:
    disp = True
else:
    disp = False
if disp:
    E_disp = temp[-7]
E_xc = temp[-6] + temp[-5]
E_NN = temp[-3]
E_J = temp[-1]
E_vpp = temp[-2]
E_KE = temp[-4]

if disp:
    E_rho_xc = E_total + E_J - E_xc - E_NN - E_eig - E_disp
else:
    E_rho_xc = E_total + E_J - E_xc - E_NN - E_eig

# if disp:
#     E_total2 = E_eig - E_J - E_rho_xc + E_NN + E_xc + E_disp
# else:
#     E_total2 = E_eig - E_J - E_rho_xc + E_NN + E_xc

if os.path.exists(folder+'/data.pkl'):
    data = pickle.load(open(folder+'/data.pkl', 'rb'))
else:
    data = {}

data['T'] = T
data['Na'] = Na
data['Nb'] = Nb
data['mua'] = mua
data['mub'] = mub
data['E_KE'] = E_KE
data['E_vpp'] = E_vpp
data['E_rho_xc'] = E_rho_xc
data['E_eig'] = E_eig
data['E_J'] = E_J
data['E_NN'] = E_NN
data['E_xc'] = E_xc
data['E_disp'] = E_disp if disp else None
data['E_total_output'] = E_total

with open(folder+'/data.pkl', 'wb') as f:
    pickle.dump(data, f)
print('mu a:', mua)
print('mu b:', mub)
print(' ')
print('Total Energy from output:', E_total)
