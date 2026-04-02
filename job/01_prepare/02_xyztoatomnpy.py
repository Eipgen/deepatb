import numpy as np
import os
from glob import glob
#from multiprocessing import Pool
import os
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="xyz file")
args = parser.parse_args()



AntoBohr=0.5291772083

ELEMENTS = ['X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
CHARGES = dict(((str(x),i) for i,x in enumerate(ELEMENTS)))


def split_file_by_empty_lines(input_filename):
    index=[]
    for line in range(len(input_filename)):  
        line1 = input_filename[line].strip()
        if not line1:
            index.append(line)
    return index

def ChooseTest(log):
    T=open(log).readlines()[2:]
    #index=split_file_by_empty_lines(T)[0]
    T=T
    xyz=[]
    for i in T:
        xyz0=i.split()[1:]
        xyz0=[float(j)/AntoBohr for j in xyz0]
        element=[CHARGES[i.split()[0]]]
        xyz.append(element+xyz0)
    xyz=np.expand_dims(np.asarray(xyz),axis=0)
    return xyz
print(CHARGES["H"])


xyz=glob(args.dir+"/*xyz")

for i in xyz:
    inpu=ChooseTest(i)
    os.system("mkdir -p npydata/"+i.split("/")[1].split(".xyz")[0])
    np.save("npydata/"+i.split("/")[1].split(".xyz")[0]+"/atom.npy",inpu)

