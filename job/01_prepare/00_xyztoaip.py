import argparse
import os
import glob

xyzfile=open("xyzfile.raw")

#xyzfile=open("xyzfile.raw").readlines()
xyzfile=[i.strip() for i in xyzfile]

def split_file_by_empty_lines(input_filename):
    index=[]
    for line in range(len(input_filename)):  
        line1 = input_filename[line].strip()
        if not line1:
            index.append(line)
    return index

def xyztoaip(xyz):
    T=open(xyz).readlines()[2:]
    xyz=T
    keywords=["! atb",">atb","repeleself on","end",">ope","out 2 ","fden off","end",">xyz 0 1"]
    xyzfile=[" ".join(i.strip().split(",")) for i in xyz]
    end=["end"]
    aip=keywords+xyzfile+end
    return aip

for i in xyzfile:
    mop=xyztoaip(i)
    name=i.split("/")[-1]
    f=open(i+".aip","w")
    f.write("\n".join(mop))
    f.close()
