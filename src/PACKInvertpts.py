import numpy as np
import h5py
import os

def Invertpts(datfol,sigthres=4):
    assert False,"Not coded"
    h5Al=h5py.File(os.path.join(datfol,"Al"),"r")
    h5Preparefn=os.path.join(datfol,"Prepare")
    if os.path.exists(h5Preparefn):
        os.remove(h5Preparefn)
    h5Prepare=h5py.File(h5Preparefn,"w")
    for key,val in h5Al.attrs.items():
        h5Prepare.attrs[key]=val

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UNet2d.Net(n_channels=1,num_classes=2)
    net.load_state_dict(torch.load("./src/NN/herm_highlighter_aug.pth"))
    net.to(device=device)
    W,H=h5Prepare.attrs["W"],h5Al.attrs["H"]
    grid=np.moveaxis(np.array(np.meshgrid(np.arange(W),np.arange(H),indexing="ij")),0,2)
    Wpad,Hpad=padto16(W,H)
    prepareset=h5Prepare.create_dataset("Al_to_Prepare",(h5Prepare.attrs["T"],2,3),dtype="f4")
    mean=h5Al.attrs["mean"]
    sig=h5Al.attrs["std"]
    thres=mean+sigthres*sig

    for i in range(h5Prepare.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5Prepare.attrs["T"]),end="")
        fr=np.pad(np.array(h5Al[str(i)+"/frame"]),((0,0),(0,Wpad),(0,Hpad),(0,0))).astype(np.int16)
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
    h5Prepare.attrs["high_exists"]=True
    #dset=h5Prepare.create_dataset("pointdat",(),dtype="i2",compression="gzip")


    h5Al.close()
    h5Prepare.close()
