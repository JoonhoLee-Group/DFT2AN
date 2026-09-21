#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 15:05:56 2025

@author: liwenko
"""

import numpy as np 
import re

Ha2eV = 27.211386245988

def read_arma_cube(filename, dtype=np.complex128):
    # read Arma binary cube
    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode().strip()
        if not header.startswith('ARMA_CUB_BIN'):
            raise ValueError('Not an Armadillo cube binary file')
    
        # Read dimensions
        dims_line = f.readline().decode().strip()
        n_rows, n_cols, n_slices = map(int, dims_line.split())
    
        # Read binary data (float64)
        data = np.fromfile(f, dtype=dtype)
    
        # Armadillo stores cubes in column-major order
        return data.reshape((n_rows, n_cols, n_slices), order='F')

def read_arma_mat(filename, dtype=np.float64):
    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode().strip()
        if not header.startswith('ARMA_MAT_BIN'):
            raise ValueError('Not an Armadillo mat binary file')

        # Read dimensions
        dims_line = f.readline().decode().strip()
        n_rows, n_cols = map(int, dims_line.split())

        # Read binary data (float64)
        data = np.fromfile(f, dtype=dtype)

        # Armadillo stores cubes in column-major order
        return data.reshape((n_rows, n_cols), order='F')

def get_EFermi(filename, unit='au'):
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith(' The final Fermi energy'):
                EF_eV = float(line.strip().split()[-2])
                if unit == 'eV':
                    return EF_eV
                elif unit == 'au':
                    return EF_eV / Ha2eV
                else: 
                    raise Exception('unrecognized unit')
            elif line.startswith(' The final alpha Fermi energy'):
                EFa_eV = float(line.strip().split()[-2])
                line = f.readline()
                line = f.readline()
                EFb_eV = float(line.strip().split()[-2])
                if unit == 'eV':
                    return EFa_eV, EFb_eV
                elif unit == 'au':
                    return EFa_eV/Ha2eV, EFb_eV/Ha2eV
                else:
                    raise Exception(('unrecognized unit'))
            line = f.readline()

def get_N(filename):
    pattern = r"There are (\d+\.\d+) alpha and (\d+\.\d+) beta electrons"
    with open(filename, 'r') as f:
        line = f.readline()
        match = re.search(pattern, line)
        while not match and line:
            line = f.readline()
            match = re.search(pattern, line)
        Na, Nb = map(float, match.groups())
    return Na, Nb

def get_nk_nmo(Fock_file):
    ''' Input the fock file from scratch. Returns nk and nmo. '''
    with open(Fock_file, 'rb') as f:
        # Read header
        f.readline()
        dims_line = f.readline().decode().strip()
        n_rows, n_cols, n_slices = map(int, dims_line.split())
        return n_slices, n_rows
    
def get_bs(outputfile, unit='au'):
    ''' unit can be 'au' or 'A_inv'. '''
    with open(outputfile, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('       Reciprocal Lattice Vectors'):
                break
            line = f.readline()
        f.readline()
        f.readline()
        line = f.readline()
        b1_A = np.array([float(x) for x in line.strip().split()[1:4]])
        line = f.readline()
        b2_A = np.array([float(x) for x in line.strip().split()[1:4]])
        line = f.readline()
        b3_A = np.array([float(x) for x in line.strip().split()[1:4]])
    if unit == 'A_inv':
        return b1_A, b2_A, b3_A
    elif unit == 'au':
        Bohr2A = 0.529177
        return Bohr2A*b1_A, Bohr2A*b2_A, Bohr2A*b3_A
    else:
        raise Exception('unrecognized unit')
        
def get_nk123(outputfile):
    ''' returns the MP mesh number of points '''
    with open(outputfile, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith(' Monkhorst-Pack mesh'):
                break
            line = f.readline()
    temp = line.strip().split()[-1]
    nk1, nk2, nk3 = [int(x) for x in temp.strip('[]').split(',')]
    return nk1, nk2, nk3

def get_eq_kpts(outputfile):
    rtn = []
    with open(outputfile, 'r') as f:
        line = f.readline()
        while line:
            if 'k-point orbital energies' in line:
                eq_pts = []
                while 'k-point' in line:
                    k_ind = int(line[4:].strip().split()[0].strip('-th'))
                    eq_pts.append(k_ind)
                    line = f.readline()
                rtn.append(eq_pts)
            line = f.readline()
    return rtn

def expand_irr_k(A_irr, eq_kpts):
    ''' A_irr is some array with the last dimension being the irreducible kpt.
    Use eq_kpts to expand A_irr into the full reducible kpts. '''
    nk = 0
    for v in eq_kpts:
        nk += len(v)
    A = np.zeros(np.shape(A_irr)[:-1]+(nk,), dtype=A_irr.dtype)
    for i in range(len(eq_kpts)):
        v = eq_kpts[i]
        for k_ind in v:
            A[..., k_ind] = A_irr[..., i]
    return A   

def read_DOS(DOS_dat_file):
    ''' Read the DOS.dat file from qcpbc. Returns Es and DOS as two arrays. '''
    DOS_full = np.loadtxt(DOS_dat_file, skiprows=1, delimiter=',')
    return DOS_full[:,0], DOS_full[:,1]

def get_totalE(outputfile, extrapolate=True):
    with open(outputfile, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith(' Final energy is'):
                E_T = float(line.strip().split()[-1])
            if line.startswith(' Free Energy (E - TS) is'):
                F_T = float(line.strip().split()[-1])
                break
            line = f.readline()
    if extrapolate:
        return (E_T+F_T)/2
    else:
        return E_T