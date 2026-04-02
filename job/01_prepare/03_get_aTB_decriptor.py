#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import glob  as glob 
import abc
import time
import torch
import numpy as np
from torch import nn
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf, dft
from ase.io import read
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="aop file dir")
args = parser.parse_args()

def extract_data(filename):
    start_marker = "Density Matrix:"
    end_marker = "Full Mulliken population analysis:"
    with open(filename, 'r') as file:
        lines = file.readlines()
    start_index = end_index = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i + 1
        if end_marker in line:
            end_index = i-1
            break
    extracted_lines = lines[start_index:end_index] if start_index and end_index else []
    return extracted_lines

def are_all_integers(lst):
    try:
        for item in lst:
            int(item)
        return True
    except ValueError:
        return False

def fill_matrix(data):
    # get matrix
    max_index = int(max([row[0] for row in data]))
    matrix=np.zeros((max_index,max_index))
    for row in data:
        i,j,value=int(row[0]),int(row[1]),row[2]
        matrix[i-1][j-1] = value
        matrix[j-1][i-1] = value
    return matrix

def give_matrix(data):
    int_index=[]
    for h in range(len(data)):
        #print(data[h])
        j=data[h].strip().split()
        if are_all_integers(j):
            int_index.append([h,j])
    num=[[len(data),"end"]]
    int_index = int_index + num
    matrix=[]
    for i in range(len(int_index)-1):
        data1=data[int_index[i][0]+1:int_index[i+1][0]]
        intnum=int_index[i][1]
        for j in data1:
            row_j=j.strip().split()
            for k in range(len(row_j[1:])):
                matrix.append([int(row_j[0]),int(intnum[k]),float(row_j[1:][k])])
    matrix=np.array(matrix)
    matrix_end = fill_matrix(matrix)
    return matrix_end

_zeta = 1.5**np.array([17,13,10,7,5,3,2,1,0,-1,-2,-3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size-1), k=1)
_table = np.concatenate([_zeta.reshape(-1,1), _coef], axis=1)
DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]

def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    elif isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    elif not isinstance(basis, str):
        return basis
    elif basis.endswith(".npy"):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    elif basis.endswith(".npz"):
        all_tables = np.load(basis)
        return [[int(name.split("_L")[-1]) if "_L" in name else ii, *table.tolist()] 
                for ii, (name, table) in enumerate(all_tables.items())]
    else:
        from pyscf import gto
        symb = DEFAULT_SYMB
        if "@" in basis:
            basis, symb = basis.split("@")
        return gto.basis.load(basis, symb=symb)

def get_shell_sec(basis):
    if not isinstance(basis, (list, tuple)):
        basis = load_basis(basis)
    shell_sec = []
    for l, c0, *cr in basis:
        nb = c0 if isinstance(c0, int) else (len(c0)-1)
        shell_sec.extend([2*l+1] * nb)
    return shell_sec

project_basis=DEFAULT_BASIS
shell_sec=get_shell_sec(project_basis)
nproj=sum(shell_sec)

def gen_proj_mol(mol, basis):
    mole_coords = mol.atom_coords(unit="Ang")
    mole_ele = mol.elements
    test_mol = gto.Mole()
    test_mol.atom = [["X", coord] for coord, ele in zip(mole_coords, mole_ele)
                     if not ele.startswith("X")]
    test_mol.basis = basis
    test_mol.build(0,0,unit="Ang")
    return test_mol


def proj_intor(intor,mol,pmol):
    """1-electron integrals between origin and projected basis"""
    proj = gto.intor_cross(intor, mol, pmol)
    return proj

def proj_ovlp(mol,pmol):
    """overlap between origin and projected basis, reshaped"""
    nao = mol.nao
    natm = pmol.natm 
    pnao = pmol.nao
    proj = proj_intor("int1e_ovlp",mol,pmol)
    # return shape [nao x natom x nproj]
    return proj.reshape(nao, natm, pnao // natm)


def prepare_integrals(mol):
    # a virtual molecule to be projected on
    pmol = gen_proj_mol(mol,project_basis)
    # < mol_ao | alpha^I_rlm >, shape=[nao x natom x nproj]
    t_proj_ovlp = torch.from_numpy(proj_ovlp(mol,pmol)).double()
    # split the projected coeffs by shell (different r and l)
    t_ovlp_shells = torch.split(t_proj_ovlp, shell_sec, -1)
    return t_ovlp_shells




def t_make_pdm(dm, ovlp_shells):
    """return projected density matrix by shell"""
    # (D^I_rl)_mm' = \sum_i < alpha^I_rlm | phi_i >< phi_i | aplha^I_rlm' >
    pdm_shells = [torch.einsum('rap,...rs,saq->...apq', po, dm, po)
                    for po in ovlp_shells]
    return pdm_shells


def t_shell_eig(pdm):
    return torch.linalg.eigvalsh(pdm)

def t_make_eig(dm, ovlp_shells):
    """return eigenvalues of projected density matrix"""
    pdm_shells = t_make_pdm(dm, ovlp_shells)
    eig_shells = [t_shell_eig(dm) for dm in pdm_shells]
    ceig = torch.cat(eig_shells, dim=-1)
    return ceig


def make_eig(dm,t_ovlp_shells):
    #return eigenvalues of projected density matrix
    t_dm = torch.from_numpy(dm).double()
    t_eig = t_make_eig(t_dm, t_ovlp_shells)
    return t_eig.detach().cpu().numpy()


xyz = open("xyzfile.raw").readlines()
xyz=[i.strip() for i in xyz]


for i in xyz:
    atoms=read(i)
    mol = gto.M(verbose = 0,atom=atoms_from_ase(atoms),basis = {
    'C': gto.basis.parse('''
  
    C    S
        3.0721175056       -0.0280509700
        1.3687758384       -0.1746882872
        0.3240771938        1.1201153994
         
         P
        2.8721629104        0.2766137964
        0.6812493146        0.6838031157
        0.1731736502        0.3919574022
    '''),
    'H': gto.basis.parse('''
    H    S
        5.2207102776        0.1405037194
        0.5049757957        1.0787664652
        0.0774854869        0.4446345270
    '''),
    'N': gto.basis.parse('''
    N    S
        4.5879484039       -0.1338512778
        1.5800601766        0.1042359172
        0.4429062122        0.7301154137
         
         P
        3.3458608336        0.2623439757
        1.5582459500        0.4014224904
        0.2880171064        0.3919574022
    '''),
    'O': gto.basis.parse('''
    O    S
        5.7040416783       -0.3630816107
        2.0696822173        0.1245625557
        0.8742035501        0.7001154423
        
         P
        5.9584000975        0.2570856555
        1.7024614553        0.2882919113
        0.4774601324        0.3919574022
    '''),
    'F': gto.basis.parse('''
    F    S
        7.4097452381       -0.3032134220
        3.7219494446        0.0857645149
        0.8236853351        0.7001154423
        
         P
        9.8630461453        0.1251103365
        1.5503718659        0.2593507468 
        0.4428702207        0.3919574022
    '''),
    'S': gto.basis.parse('''
    S   S
        4.4391218578       -0.3846518685
        0.5714637490        0.2723396978
        0.3241647738        0.9003984332
        
        P
        0.8524431388        0.0042142171
        0.2277542609        0.3375041334
        0.2105670005        0.4620010257
        
        D
        0.7150274233        1.0000000000
    '''),
    'Cl': gto.basis.parse('''
    Cl   S
        6.0668586306       -0.3491429343
        1.0216169791        0.1046966664
        0.3369340209        0.9003984332
         
         P
        1.8620014729        0.0023849980
        0.3309623820        0.6244837006
        0.1568383164        0.4620010257
         
         D
        0.9763895683        1.0000000000
    ''')})
    t_ovlp_shells=prepare_integrals(mol)
    j=i.split("/")[-1]
    data = extract_data(args.dir+"/"+j+".aop")
    data = [i.strip() for i in data]
    dm=give_matrix(data)
    dm_eig=make_eig(dm,t_ovlp_shells)
    dm_eig=np.expand_dims(dm_eig,axis=0)
    path="npydata/"+i.split("/")[1].split(".")[0]
    os.system("mkdir -p "+path)
    np.save(path+"/dm_eig.npy",dm_eig)
    nao = mol.nao
    natm = mol.natm 
    pnao = mol.nao
    nproj=108
    H=np.array([natm,natm,nao,nproj])    
    np.savetxt(path+"/system.raw",H.reshape(1, -1),comments="# natom natom_raw nao nproj",delimiter=" ",fmt="%d",newline='')
    print(i,dm_eig.shape)
