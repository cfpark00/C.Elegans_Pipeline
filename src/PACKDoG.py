import numpy as np
import h5py
import os
import multiprocessing
import cv2

def DoG_one(pack):
    im,s1,s2,S=pack
    for z in range(im.shape[3]):
        img=np.moveaxis(im[:,:,:,z],0,2)#to WHC
        imgs1=cv2.GaussianBlur(img,(S,S),s1,borderType=cv2.BORDER_REPLICATE)
        imgs2=cv2.GaussianBlur(img,(S,S),s2,borderType=cv2.BORDER_REPLICATE)
        im[:,:,:,z]=np.moveaxis(imgs1-imgs2,2,0)#to CWH
    im=np.clip(im,0,1)#this is not supposed to do much, shouldn't cause saturation
    im=(im*255).astype(np.int16)
    return im

def DoG(datfol,s1=1,s2=3,S=21):
    #file handling

    h5dogfn=os.path.join(datfol,"DoG")
    if os.path.exists(h5dogfn):
        os.remove(h5dogfn)

    h5raw=h5py.File(os.path.join(datfol,"h5raw"),"r")
    h5dog=h5py.File(h5dogfn,"w")

    #meta data handling
    for key,val in h5raw.attrs.items():
        h5dog.attrs[key]=val
    sh=(h5raw.attrs["C"],h5raw.attrs["W"],h5raw.attrs["H"],h5raw.attrs["D"])
    minarr=np.array([h5raw.attrs["rmin"],h5raw.attrs["gmin"]])
    maxarr=np.array([h5raw.attrs["rmax"],h5raw.attrs["gmax"]])

    #start
    for i in range(h5raw.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5raw.attrs["T"]),end="")
        imfloat=(np.array(h5raw[str(i)+"/frame"])-minarr[:,None,None,None])/(maxarr-minarr)[:,None,None,None]
        im=DoG_one((imfloat,s1,s2,S))
        dset=h5dog.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        dset[...]=im

    h5dog.close()
    h5raw.close()






















########################TRASH##############
"""
def get_dog_filter(s1,s2,S):
    assert S%2==1
    cent=S//2
    grid=np.array(np.meshgrid(np.arange(S),np.arange(S),indexing="ij"))
    s1filt=np.exp(-np.sum(np.square(grid-cent),axis=0)/(2*s1**2))/(2*np.pi*s1**2)
    s2filt=np.exp(-np.sum(np.square(grid-cent),axis=0)/(2*s2**2))/(2*np.pi*s2**2)
    filt=s1filt-s2filt
    return filt/np.sum(filt)

def DoG_one(pack):
    im,filt=pack
    for c in range(im.shape[0]):
        for z in range(im.shape[3]):
            im[c,:,:,z]=sim.convolve(im[c,:,:,z],filt,mode="constant")
    mean=np.mean(im,axis=(1,2,3))
    std=np.std(im,axis=(1,2,3))
    thres=mean+1*std
    im[0]=np.clip(im[0],thres[0],np.inf)-thres[0]
    im[1]=np.clip(im[1],thres[1],np.inf)-thres[1]
    im=(im*255).astype(np.int16)
    return im

def DoG(datfol,chunksize=20):
    h5raw=h5py.File(os.path.join(datfol,"h5raw"),"r")
    h5dogfn=os.path.join(datfol,"DoG")
    if os.path.exists(h5dogfn):
        os.remove(h5dogfn)
    h5dog=h5py.File(h5dogfn,"w")
    for key,val in h5raw.attrs.items():
        h5dog.attrs[key]=val
    sh=(h5raw.attrs["C"],h5raw.attrs["W"],h5raw.attrs["H"],h5raw.attrs["D"])
    minarr=np.array([h5raw.attrs["rmin"],h5raw.attrs["gmin"]])
    maxarr=np.array([h5raw.attrs["rmax"],h5raw.attrs["gmax"]])
    print(minarr,maxarr)
    filt=get_dog_filter(1,2,21)
    for i in range(h5raw.attrs["T"]):
        print(i)
        imfloat=(np.array(h5raw[str(i)+"/frame"])-minarr[:,None,None,None])/(maxarr-minarr)[:,None,None,None]
        print(np.min(imfloat),np.max(imfloat))
        im=DoG_one((imfloat,filt.copy()))
        print(np.min(im),np.max(im))
        dset=h5dog.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        print(im.shape)
        dset[...]=im
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(np.max(im[0],axis=2).T)
        plt.subplot(122)
        plt.imshow(np.max(im[1],axis=2).T)
        plt.show()
        break
    h5dog.close()
    h5raw.close()
"""

"""
def DoG(datfol,gpu=False,chunksize=20):
    try:
        import torch
        if not torch.cuda.is_available():
            gpu=False
    except:
        gpu=False

    ####
    h5raw=h5py.File(os.path.join(datfol,"h5raw"),"r")
    h5dogfn=os.path.join(datfol,"DoG")
    if os.path.exists(h5dogfn):
        os.remove(h5dogfn)
    h5dog=h5py.File(h5dogfn,"w")
    for key,val in h5raw.attrs.items():
        h5dog.attrs[key]=val
    sh=(h5raw.attrs["C"],h5raw.attrs["W"],h5raw.attrs["H"],h5raw.attrs["D"])
    if gpu:
        assert False, "Not Implemented"
        nproc=batch
        filt=torch.tensor(filt).to(device="cuda",dtype=torch.float32)
        for i in range(h5raw.attrs["T"]//nproc):
            print("\r\t"+str(nproc*i)+"/"+str(h5raw.attrs["T"]),end="")
            with torch.no_grad():
                ims=torch.tensor(np.array([h5raw[str(j)] for j in range(i*nproc,(i+1)*nproc)])/255).to(device="cuda",dtype=torch.float32)
                ims=ims.permute(0,4,1,2,3)
                ims=ims.reshape(-1,ims.size(2),ims.size(3),ims.size(4))
                ims=torch.nn.functional.conv2d(ims,groups=ims.size(0))
                ims=ims.reshape(*shtemp,ims.size(1),ims.size(2),ims.size(3)).permute(0,2,3,4,1).cpu().detach().numpy()
            for j in range(nproc):
                dset=h5dog.create_dataset(str(i*nproc+j)+"/frame",sh,dtype="i2",compression="gzip")
                dset[...]=ims[j]
        with torch.no_grad():
            ims=torch.tensor(np.array([h5raw[str(j)] for j in range(nproc*(h5raw.attrs["T"]//nproc),h5raw.attrs["T"])])/255).to(device="cuda",dtype=torch.float32)
            ims=ims.permute(0,4,1,2,3)
            shtemp=(ims.size(0),ims.size(1))
            ims=ims.reshape(-1,ims.size(2),ims.size(3),ims.size(4))
            ims=torch.nn.functional.conv2d(ims,groups=ims.size(0))
            ims=ims.reshape(*shtemp,ims.size(1),ims.size(2),ims.size(3)).permute(0,2,3,4,1).cpu().detach().numpy()

        dset=h5dog.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        dset[...]=im
    else:
        if False:#multiprocessing
            minarr=np.array([h5raw.attrs["rmin"],h5raw.attrs["gmin"]])
            maxarr=np.array([h5raw.attrs["rmax"],h5raw.attrs["gmax"]])

            def pack(j):
                print(j)
                #0~1 norm
                return ((np.array(h5raw[str(j)+"/frame"])-minarr[:,None,None,None])/(maxarr-minarr)[:,None,None,None],filt.copy())

            p=multiprocessing.Pool(multiprocessing.cpu_count())
            res=p.imap(DoGone,(pack(i) for i in range(h5raw.attrs["T"])),chunksize=chunksize)

            for idx,im in enumerate(res):
                print(str(idx))
                dset=h5dog.create_dataset(str(idx)+"/frame",sh,dtype="i2",compression="gzip")
                dset[...]=im
        minarr=np.array([h5raw.attrs["rmin"],h5raw.attrs["gmin"]])
        maxarr=np.array([h5raw.attrs["rmax"],h5raw.attrs["gmax"]])
        print(minarr,maxarr)
        filt=get_dog_filter(1,2,21)
        for i in range(h5raw.attrs["T"]):
            print(i)
            imfloat=(np.array(h5raw[str(i)+"/frame"])-minarr[:,None,None,None])/(maxarr-minarr)[:,None,None,None]
            print(np.min(imfloat),np.max(imfloat))
            im=DoGone((imfloat,filt.copy()))
            print(np.min(im),np.max(im))
            dset=h5dog.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
            print(im.shape)
            dset[...]=im
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(np.max(im[0],axis=2).T)
            plt.subplot(122)
            plt.imshow(np.max(im[1],axis=2).T)
            plt.show()
            break
    h5dog.close()
    h5raw.close()
"""
