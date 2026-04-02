import numpy as np
import torch
from deepks.model import CorrNet
from deepks.utils import load_yaml
import argparse


parser = argparse.ArgumentParser("get deepatb output energy")
parser.add_argument("--model",type=str, default="model.pth", help="path of model")
parser.add_argument("--raw",type=str, default="valid.raw", help="path of dm_eig npydata raw")

args = parser.parse_args()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

model = CorrNet.load_dict(torch.load(args.model, map_location=device))
#print(torch.load(args.model, map_location=device))
elem_dict=model.elem_dict
B=elem_dict
print(B)
#B = np.array([[key, value] for key, value in elem_dict.items()])
model.eval()

ENE_tot=[]
data=open(args.raw).readlines()
for i in data:
    i=i.strip()
    #sample_feature = np.load(args.dir+"/dm_eig.npy")
    sample_feature = np.load(i+"/dm_eig.npy")
    sample_tensor = torch.from_numpy(sample_feature).float()
    with torch.no_grad():
        energy_correction = model(sample_tensor).item()
    atom=np.load(i+"/atom.npy")[0][:,0]
    atom_E=[]
    for at in atom:
        if at==1:
            atom_E.append(B[1])
        elif at==3:
            atom_E.append(B[3])
        elif at==5:
            atom_E.append(B[5])
        elif at==6:
            atom_E.append(B[6])
        elif at==7:
            atom_E.append(B[7])
        elif at==8:
            atom_E.append(B[8])
        elif at==9:
            atom_E.append(B[9])
        elif at==11:
            atom_E.append(B[11])
        elif at==12:
            atom_E.append(B[12])
        elif at==14:
            atom_E.append(B[14])
        elif at==15:
            atom_E.append(B[15])
        elif at==16:
            atom_E.append(B[16])
        elif at==17:
            atom_E.append(B[17])
        elif at==19:
            atom_E.append(B[19])
        elif at==20:
            atom_E.append(B[20])
        elif at==35:
            atom_E.append(B[35])
        elif at==53:
            atom_E.append(B[53])
    base_energy = np.load(i+"/e_base.npy")[0]
    l_e_delta = np.load(i+"/l_e_delta.npy")[0][0]
    E_atom=sum(atom_E) ## sum of atomization energy
    #print(i.split("/")[-1],base_energy,l_e_delta, energy_correction,E_atom)
    #print(i.split("/")[-1],base_energy,base_energy+l_e_delta,base_energy+energy_correction*0.5+E_atom,abs((base_energy+l_e_delta)-(base_energy+energy_correction+E_atom)))
    print(i.split("/")[-1],base_energy+energy_correction+E_atom)
