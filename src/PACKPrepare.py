import numpy as np
import h5py
import os
import scipy.ndimage as sim
import cv2
import shutil

def get_cent_pts(high,wormthres,num_segments,grid):
    wormmask=np.sum(high,axis=0)>wormthres
    labeled, nr_objects=sim.label(wormmask)
    _,counts=np.unique(labeled,return_counts=True)
    if len(counts)<2:#only background
        return None
    wormmask=(labeled==(np.argmax(counts[1:])+1))

    red_mask=high[0]/255
    green_mask=high[1]/255

    ratiomask=2*wormmask+wormmask*(red_mask-green_mask)#1 to 3
    allvals=ratiomask[wormmask]
    segment_lims=np.min(allvals),np.max(allvals)
    segments=np.linspace(segment_lims[0]-1e-8,segment_lims[1]+1e-8,num_segments+1)
    pts=[]
    segs=[]
    for i in range(num_segments):
        segmask=(segments[i]<=ratiomask)&(ratiomask<segments[i+1])
        segs.append(segmask)
        norm_s=np.sum(segmask)
        if norm_s==0:
            return None
        else:
            pts.append(np.sum(grid*segmask[:,:,None],axis=(0,1))/norm_s)
    return np.array(pts)

def extract_affine(pts,W,H):
    if pts is None:
        return np.array([[1,0,0],[0,1,0]]).astype(np.float32)
    cm=np.array([np.mean(pts,axis=0)])
    pts=pts-cm
    stds=np.std(pts,axis=0)
    if stds[0]>stds[1]:
        coeffs=np.polyfit(pts[:,0],pts[:,1],1)#y=ax+b
        inv=False
        if pts[0,0]>pts[-1,0]:
            inv=True
        theta=np.arctan(coeffs[0])+inv*np.pi
    else:
        coeffs=np.polyfit(pts[:,1],pts[:,0],1)#x=ay+b
        inv=False
        if pts[0,1]>pts[-1,1]:
            inv=True
        theta=np.arctan(coeffs[0])+inv*np.pi
        theta=np.pi/2-theta

    rotmat=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    xtrans,ytrans=(np.matmul(rotmat,cm.T).T)[0]
    affmat=np.array([[np.cos(theta),np.sin(theta),W//2-xtrans],[-np.sin(theta),np.cos(theta),H//2-ytrans]])
    return affmat.astype(np.float32)

def affine_transform_midprepare(fr,high,affmat,nocut=False):
    #print(affmat)
    #affmat=np.array([[1,0,0],[0,1,0]]).astype(np.float32)
    imr=fr[0]
    img=fr[1]
    if high is not None:
        high=np.moveaxis(high,0,2)#rg becomes channel
    sh=tuple(imr.shape[:2])
    imr=cv2.warpAffine(imr.swapaxes(0,1), affmat,sh).swapaxes(0,1)
    img=cv2.warpAffine(img.swapaxes(0,1), affmat,sh).swapaxes(0,1)
    if high is not None:
        high=cv2.warpAffine(high.swapaxes(0,1), affmat,sh).swapaxes(0,1)
    W,H,D=imr.shape
    imr=sim.zoom(imr,(1,1,16/D),order=1)
    img=sim.zoom(img,(1,1,16/D),order=1)
    im=np.concatenate([imr[None,...],img[None,...]],axis=0)
    if nocut:
        wl,wh=0,W
        hl,hh=0,H
    else:
        wl,wh=W//2-128,W//2+128
        hl,hh=H//2-80,H//2+80
    if high is None:
        return im[:,wl:wh,hl:hh,:]
    return im[:,wl:wh,hl:hh,:],np.moveaxis(high[wl:wh,hl:hh,:],2,0)

def padto16(W,H):
    Wpad=0
    Hpad=0
    if W%16!=0:
        Wpad=16-W%16
    if H%16!=0:
        Hpad=16-H%16
    return Wpad,Hpad

def refine(high):
    wormint=np.sum(high,axis=0)
    high=np.where(wormint>0.5,high/wormint,0)
    return high

def Prepare(datfol,sigthres=4):
    from src.NN import UNet2d
    import torch
    h5Al=h5py.File(os.path.join(datfol,"Al"),"r")
    h5Preparefn=os.path.join(datfol,"Prepare")
    if os.path.exists(h5Preparefn):
        os.remove(h5Preparefn)
    h5Prepare=h5py.File(h5Preparefn,"w")
    for key,val in h5Al.attrs.items():
        h5Prepare.attrs[key]=val

    W,H=h5Prepare.attrs["W"],h5Al.attrs["H"]


    W,H=h5Prepare.attrs["W"],h5Al.attrs["H"]
    grid=np.moveaxis(np.array(np.meshgrid(np.arange(W),np.arange(H),indexing="ij")),0,2)
    Wpad,Hpad=padto16(W,H)
    prepareset=h5Prepare.create_dataset("Al_to_Prepare",(h5Prepare.attrs["T"],2,3),dtype="f4")
    affmat_given=False
    if "Al_to_Prepare" in h5Al.keys():
        affmat_given=True
        affmats=np.array(h5Al["Al_to_Prepare"])
        prepareset[...]=affmats.astype(np.float32)

    mean=h5Al.attrs["mean"]
    sig=h5Al.attrs["std"]
    thres=mean+sigthres*sig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UNet2d.Net(n_channels=1,num_classes=2)
    net.load_state_dict(torch.load("./src/NN/male_highlighter.pth", map_location=device))
    net.to(device=device)
    for i in range(h5Prepare.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5Prepare.attrs["T"]),end="")
        fr=np.pad(np.array(h5Al[str(i)+"/frame"]),((0,0),(0,Wpad),(0,Hpad),(0,0))).astype(np.int16)
        if affmat_given:
            fr=affine_transform_midprepare((fr[:,:W,:H,:]/255).astype(np.float32),None,affmats[i].copy(),nocut=True)
            fr=np.clip(fr*255,0,255)
            fr=(fr*(fr>thres[:,None,None,None])).astype(np.int16)#thresholding
            dset=h5Prepare.create_dataset(str(i)+"/frame",fr.shape,dtype="i2",compression="gzip")
            dset[...]=fr
            continue

        with torch.no_grad():
            high=torch.sigmoid(net(torch.tensor([[np.max(fr[0]/255,axis=2)]]).to(device=device, dtype=torch.float32))).cpu().detach().numpy()
        high=high[0][:,:W,:H]#(2,W,H)
        high=refine(high)

        pts=get_cent_pts(high,wormthres=0.5,num_segments=5,grid=grid)
        affmat=extract_affine(pts,W,H)
        fr,high=affine_transform_midprepare((fr[:,:W,:H,:]/255).astype(np.float32),high,affmat.copy(),nocut=True)
        fr=np.clip(fr*255,0,255)
        fr=(fr*(fr>thres[:,None,None,None])).astype(np.int16)#thresholding
        high=np.clip(high*255,0,255).astype(np.int16)

        dset=h5Prepare.create_dataset(str(i)+"/frame",fr.shape,dtype="i2",compression="gzip")
        dset[...]=fr
        dset=h5Prepare.create_dataset(str(i)+"/high",high.shape,dtype="i2",compression="gzip")
        dset[...]=high
        prepareset[i]=affmat

    h5Prepare.attrs["W"]=256
    h5Prepare.attrs["H"]=160
    h5Prepare.attrs["D"]=16
    if not affmat_given:
        h5Prepare.attrs["high_exists"]=True
    #dset=h5Prepare.create_dataset("pointdat",(),dtype="i2",compression="gzip")


    h5Al.close()
    h5Prepare.close()

    shutil.copyfile(h5Preparefn,os.path.join(datfol,os.path.split(datfol)[-1]+".h5"))

def Prepare_Dipole(datfol,fromfile=None,sigthres=4):
    if fromfile is None:
        h5Al=h5py.File(os.path.join(datfol,"Al"),"r")
    else:
        h5Al=h5py.File(fromfile,"r")
    h5Preparefn=os.path.join(datfol,"Prepare")
    if os.path.exists(h5Preparefn):
        os.remove(h5Preparefn)
    h5Prepare=h5py.File(h5Preparefn,"w")
    for key,val in h5Al.attrs.items():
        h5Prepare.attrs[key]=val

    W,H=h5Prepare.attrs["W"],h5Al.attrs["H"]


    W,H=h5Prepare.attrs["W"],h5Al.attrs["H"]
    grid=np.moveaxis(np.array(np.meshgrid(np.arange(W),np.arange(H),indexing="ij")),0,2)
    Wpad,Hpad=padto16(W,H)
    prepareset=h5Prepare.create_dataset("Al_to_Prepare",(h5Prepare.attrs["T"],2,3),dtype="f4")
    affmat_given=False
    if "Al_to_Prepare" in h5Al.keys():
        affmat_given=True
        affmats=np.array(h5Al["Al_to_Prepare"])
        prepareset[...]=affmats.astype(np.float32)

    mean=h5Al.attrs["mean"]
    sig=h5Al.attrs["std"]
    thres=mean+sigthres*sig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UNet2d.Net(n_channels=1,num_classes=2)
    net.load_state_dict(torch.load("./src/NN/herm_highlighter_aug.pth", map_location=device))
    net.to(device=device)
    for i in range(h5Prepare.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5Prepare.attrs["T"]),end="")
        fr=np.pad(np.array(h5Al[str(i)+"/frame"]),((0,0),(0,Wpad),(0,Hpad),(0,0))).astype(np.int16)
        if affmat_given:
            fr=affine_transform_midprepare((fr[:,:W,:H,:]/255).astype(np.float32),None,affmats[i].copy())
            fr=np.clip(fr*255,0,255)
            fr=(fr*(fr>thres[:,None,None,None])).astype(np.int16)#thresholding
            dset=h5Prepare.create_dataset(str(i)+"/frame",fr.shape,dtype="i2",compression="gzip")
            dset[...]=fr
            continue

        with torch.no_grad():
            high=torch.sigmoid(net(torch.tensor([[np.max(fr[0]/255,axis=2)]]).to(device=device, dtype=torch.float32))).cpu().detach().numpy()
        high=high[0][:,:W,:H]#(2,W,H)
        high=refine(high)

        pts=get_cent_pts(high,wormthres=0.5,num_segments=5,grid=grid)
        affmat=extract_affine(pts,W,H)
        fr,high=affine_transform_midprepare((fr[:,:W,:H,:]/255).astype(np.float32),high,affmat.copy())
        fr=np.clip(fr*255,0,255)
        fr=(fr*(fr>thres[:,None,None,None])).astype(np.int16)#thresholding
        high=np.clip(high*255,0,255).astype(np.int16)

        dset=h5Prepare.create_dataset(str(i)+"/frame",fr.shape,dtype="i2",compression="gzip")
        dset[...]=fr
        dset=h5Prepare.create_dataset(str(i)+"/high",high.shape,dtype="i2",compression="gzip")
        dset[...]=high
        prepareset[i]=affmat

    h5Prepare.attrs["W"]=256
    h5Prepare.attrs["H"]=160
    h5Prepare.attrs["D"]=16
    if not affmat_given:
        h5Prepare.attrs["high_exists"]=True
    #dset=h5Prepare.create_dataset("pointdat",(),dtype="i2",compression="gzip")


    h5Al.close()
    h5Prepare.close()

    shutil.copyfile(h5Preparefn,os.path.join(datfol,os.path.split(datfol)[-1]+".h5"))
