import numpy as np
G=open("MP2_and_aTB_energy.txt").readlines()
h0,h1=float(G[0].strip().split()[1]),float(G[0].strip().split()[2])
for h in G:
    name=h.strip().split()[0].split(".xyz")[0]
    delta_E=(float(h.strip().split()[1])/627.51)-float(h.strip().split()[2])
    e_tot=np.expand_dims(np.array(float(h.strip().split()[1])/627.51),axis=0)
    e_base=np.expand_dims(np.array(float(h.strip().split()[2])),axis=0)
    delta_E=np.expand_dims(np.array([delta_E]),axis=0)
    np.save("npydata/"+name+"/l_e_delta.npy",delta_E)
    np.save("npydata/"+name+"/e_base.npy",e_base)
    np.save("npydata/"+name+"/e_tot.npy",e_tot)
    print(name,e_tot[0],e_base[0],delta_E[0][0])
