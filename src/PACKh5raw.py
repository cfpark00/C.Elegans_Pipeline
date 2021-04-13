import numpy as np
import h5py
import os
import scipy.io as sio

def h5raw(datfol):
    datname=os.path.split(datfol)[-1]

    meta_raw=sio.loadmat(os.path.join(datfol,datname+".mat"))
    meta={}
    meta["C"]=meta_raw["C"][0,0]
    meta["W"]=meta_raw["W"][0,0]
    meta["D"]=meta_raw["D"][0,0]
    meta["H"]=meta_raw["H"][0,0]
    meta["GFP"]=meta_raw["GFP"][0,0]
    meta["RFP"]=meta_raw["RFP"][0,0]
    meta["FrameRate"]=meta_raw["FrameRate"][0,0]
    meta["ts"]=meta_raw["times"][0]
    meta["T"]=len(meta["ts"])
    meta["stage"]=meta_raw["stage"][0]

    sh=(meta["C"],meta["W"],meta["H"],meta["D"])
    size=np.prod(sh)
    size_ts=sh[0]*sh[3]*2

    if os.path.exists(os.path.join(datfol,"h5raw")):
        os.remove(os.path.join(datfol,"h5raw"))
    h5=h5py.File(os.path.join(datfol,"h5raw"),"w")
    for key,val in meta.items():
        h5.attrs[key]=val
    h5.attrs["name"]=datname
    f = open(os.path.join(datfol,datname+".bin"), "rb")
    globalminr=np.inf
    globalmaxr=-np.inf
    globalming=np.inf
    globalmaxg=-np.inf
    """
    f.seek(0, os.SEEK_SET)
    i=0
    while True:
        if i==meta["T"]:
            break
        im=np.fromfile(f,count=size, dtype=np.uint8)
        minr,ming=np.min(im,axis=(1,2,3))
        maxr,maxg=np.max(im,axis=(1,2,3))
        globalminr=min(globalminr,minr)
        globalming=min(globalming,ming)
        globalmaxr=max(globalmaxr,maxr)
        globalmaxg=max(globalmaxg,maxg)
        x=np.fromfile(f,count=size_ts, dtype=np.float64)
        i+=1
    """
    f.seek(0, os.SEEK_SET)
    i=0
    while True:
        print("\r\t"+str(i)+"/"+str(meta["T"]),end="")
        if i==meta["T"]:
            break
        im=np.fromfile(f,count=size, dtype=np.uint8)
        im=np.transpose(im.reshape(sh[3],sh[0],sh[2],sh[1]),(1,3,2,0))
        im[1]=np.flip(im[1],axis=0)
        im=im.astype(np.int16)
        minr,ming=np.min(im,axis=(1,2,3))
        maxr,maxg=np.max(im,axis=(1,2,3))
        globalminr=min(globalminr,minr)
        globalming=min(globalming,ming)
        globalmaxr=max(globalmaxr,maxr)
        globalmaxg=max(globalmaxg,maxg)

        dset=h5.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        dset[...]=im
        x=np.fromfile(f,count=size_ts, dtype=np.float64)
        i+=1
    f.close()

    h5.attrs["gmin"]=globalming
    h5.attrs["gmax"]=globalmaxg
    h5.attrs["rmin"]=globalminr
    h5.attrs["rmax"]=globalmaxr
    h5.close()
    if i!=meta["T"]:
        print("WARNING: TIME NOT MATCHING")
