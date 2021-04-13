import numpy as np
import h5py
import os
import json
from . import GetPtsWidget
from PyQt5.QtWidgets import *
import sys
import cv2
import scipy.optimize as sopt

def get_opt_align(ptsfrom,ptsto):
    cmfrom=np.mean(ptsfrom,axis=0)
    cmto=np.mean(ptsto,axis=0)
    almat=np.array([[1,0,cmto[0]-cmfrom[0]],[0,1,cmto[1]-cmfrom[1]]])
    def energy(almat,ptsfrom,ptsto):
        almat=almat.reshape(2,3)
        return np.sum(np.square(np.matmul(almat[:,:2],ptsfrom.T).T+almat[:,2][None,:]-ptsto))
    res=sopt.minimize(energy,almat.flatten(),args=(ptsfrom,ptsto),method = 'SLSQP')
    if not res.success:
        return None
    almat=res.x.reshape(2,3)
    #print(almat)
    return almat
def pmm(im):
    print(np.min(im),np.max(im))

def apply_affine(im3d,almat):
    im3d=im3d.swapaxes(0,1)#cv2 convention
    #almat[:,:2]=almat[:,:2]
    #almat=np.array([[1,0,0],[0,1,0]]).astype(np.float32)
    return cv2.warpAffine(im3d, almat,tuple(np.array(im3d.shape)[[1,0]])).swapaxes(0,1)#z is treated as channel

#make and check alignment
def Al(datfol):
    h5Alfn=os.path.join(datfol,"Al")
    if os.path.exists(h5Alfn):
        os.remove(h5Alfn)
    h5DoG=h5py.File(os.path.join(datfol,"DoG"),"r+")

    if "almat" not in h5DoG.attrs.keys():
        ptsfrom=[]
        ptsto=[]
        #####Extract points GUI
        while True:
            comm=input("Alignment[0,"+str(h5DoG.attrs["T"])+"]:")
            try:
                comm=int(comm)
            except:
                pass
            if type(comm)==int:
                if not (0<=comm<h5DoG.attrs["T"]):
                    comm=np.random.randint(0,h5DoG.attrs["T"])
                print("Opening: "+str(comm))
                im=np.max(h5DoG[str(comm)+"/frame"],axis=3)#CWHD
                app=QApplication(sys.argv)
                diag = GetPtsWidget.GetPtsWidget("Get Points: "+str(comm)+"/"+str(h5DoG.attrs["T"]),im[0],im[1],ptsfrom,ptsto)
                app.exec_()
                del app
                print("Current number of points: ",len(ptsfrom))
                continue
            if comm=="Done":
                break
            else:
                print("Alignment[0,"+str(h5DoG.attrs["T"])+"]:","Enter a number or \"Done\"")
                continue
        if len(ptsfrom)!=len(ptsto) or len(ptsto)<3:
            assert False, "Aborting: at least 3 points needed."
        h5DoG.attrs["sugg_affpts"]=np.array([ptsfrom,ptsto])#2xNx2 point save
        print("Suggested Points Saved")

        ##Get affine transform
        almat=get_opt_align(np.array(ptsfrom),np.array(ptsto))
        if almat is None:
            print("Alignment Extraction Failed")
            h5DoG.close()
            return
        else:
            print("Alignment Extraction Successful")
        almat=almat.astype(np.float32)

        ##Check if Affine transform is good
        if True:
            result=[False]
            app=QApplication(sys.argv)
            num=6
            shows=np.random.choice(h5DoG.attrs["T"],num,replace=False)
            ims={}
            for i in shows:
                imal=np.array(h5DoG[str(i)+"/frame"])
                imal[0]=apply_affine(imal[0],almat).astype(np.int16)
                ims[str(i)+"/"+str(h5DoG.attrs["T"])]=np.max(np.moveaxis(np.array([imal[0],imal[1],imal[0]]),0,3),axis=2)
            confirm=GetPtsWidget.ConfirmAlWidget("Confirm Alignment",ims,result)
            app.exec_()
            del app
            if not result[0]:
                print("Alignment Aborted")
                h5DoG.close()
                return
    else:
        almat=np.array(h5DoG.attrs["almat"]).astype(np.float32)

    #Get number of neurons
    while True:
        N_neurons=input("Number of Neurons:")
        try:
            N_neurons=int(N_neurons)
            if not (0<N_neurons<400):
                assert False
            break
        except:
            pass

    #apply affine transform
    h5Al=h5py.File(h5Alfn,"w")
    for key,val in h5DoG.attrs.items():
        h5Al.attrs[key]=val
    h5Al.attrs["N_neurons"]=N_neurons
    h5Al.attrs["almat"]=almat.astype(np.float32)

    sh=(h5Al.attrs["C"],h5Al.attrs["W"],h5Al.attrs["H"],h5Al.attrs["D"])
    mean=np.zeros(2,dtype=np.float32)
    std=np.zeros(2,dtype=np.float32)
    for i in range(h5Al.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5Al.attrs["T"]),end="")
        dset=h5Al.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        imal=np.array(h5DoG[str(i)+"/frame"])
        imal[0]=apply_affine(imal[0],almat).astype(np.int16)
        dset[...]=imal
        mean+=np.mean(imal,axis=(1,2,3))
        std+=np.mean(imal**2,axis=(1,2,3))
    mean/=h5Al.attrs["T"]
    std/=h5Al.attrs["T"]
    std-=mean**2
    std=np.sqrt(std)
    h5Al.attrs["mean"]=mean
    h5Al.attrs["std"]=std
    #print(mean,std)
    h5Al.close()
    h5DoG.close()

def Alfromfile(datfol,filename):
    h5Alfn=os.path.join(datfol,"Al")
    if os.path.exists(h5Alfn):
        os.remove(h5Alfn)
    h5DoG=h5py.File(filename,"r+")

    if "almat" not in h5DoG.attrs.keys():
        ptsfrom=[]
        ptsto=[]
        #####Extract points GUI
        while True:
            comm=input("Alignment[0,"+str(h5DoG.attrs["T"])+"]:")
            try:
                comm=int(comm)
            except:
                pass
            if type(comm)==int:
                if not (0<=comm<h5DoG.attrs["T"]):
                    comm=np.random.randint(0,h5DoG.attrs["T"])
                print("Opening: "+str(comm))
                im=np.max(h5DoG[str(comm)+"/frame"],axis=3)#CWHD
                app=QApplication(sys.argv)
                diag = GetPtsWidget.GetPtsWidget("Get Points: "+str(comm)+"/"+str(h5DoG.attrs["T"]),im[0],im[1],ptsfrom,ptsto)
                app.exec_()
                del app
                print("Current number of points: ",len(ptsfrom))
                continue
            if comm=="Done":
                break
            else:
                print("Alignment[0,"+str(h5DoG.attrs["T"])+"]:","Enter a number or \"Done\"")
                continue
        if len(ptsfrom)!=len(ptsto) or len(ptsto)<3:
            assert False, "Aborting: at least 3 points needed."
        h5DoG.attrs["sugg_affpts"]=np.array([ptsfrom,ptsto])#2xNx2 point save
        print("Suggested Points Saved")

        ##Get affine transform
        almat=get_opt_align(np.array(ptsfrom),np.array(ptsto))
        if almat is None:
            print("Alignment Extraction Failed")
            h5DoG.close()
            return
        else:
            print("Alignment Extraction Successful")
        almat=almat.astype(np.float32)

        ##Check if Affine transform is good
        if True:
            result=[False]
            app=QApplication(sys.argv)
            num=6
            shows=np.random.choice(h5DoG.attrs["T"],num,replace=False)
            ims={}
            for i in shows:
                imal=np.array(h5DoG[str(i)+"/frame"])
                imal[0]=apply_affine(imal[0],almat).astype(np.int16)
                ims[str(i)+"/"+str(h5DoG.attrs["T"])]=np.max(np.moveaxis(np.array([imal[0],imal[1],imal[0]]),0,3),axis=2)
            confirm=GetPtsWidget.ConfirmAlWidget("Confirm Alignment",ims,result)
            app.exec_()
            del app
            if not result[0]:
                print("Alignment Aborted")
                h5DoG.close()
                return
    else:
        almat=np.array(h5DoG.attrs["almat"]).astype(np.float32)

    #Get number of neurons
    while True:
        N_neurons=input("Number of Neurons:")
        try:
            N_neurons=int(N_neurons)
            if not (0<N_neurons<400):
                assert False
            break
        except:
            pass

    #apply affine transform
    h5Al=h5py.File(h5Alfn,"w")
    for key,val in h5DoG.attrs.items():
        h5Al.attrs[key]=val
    h5Al.attrs["N_neurons"]=N_neurons
    h5Al.attrs["almat"]=almat.astype(np.float32)
    sh=(h5Al.attrs["C"],h5Al.attrs["W"],h5Al.attrs["H"],h5Al.attrs["D"])
    mean=np.zeros(2,dtype=np.float32)
    std=np.zeros(2,dtype=np.float32)
    for i in range(h5Al.attrs["T"]):
        print("\r\t"+str(i)+"/"+str(h5Al.attrs["T"]),end="")
        dset=h5Al.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="gzip")
        imal=np.array(h5DoG[str(i)+"/frame"])
        imal[0]=apply_affine(imal[0],almat).astype(np.int16)
        dset[...]=imal
        mean+=np.mean(imal,axis=(1,2,3))
        std+=np.mean(imal**2,axis=(1,2,3))
    mean/=h5Al.attrs["T"]
    std/=h5Al.attrs["T"]
    std-=mean**2
    std=np.sqrt(std)
    h5Al.attrs["mean"]=mean
    h5Al.attrs["std"]=std
    #print(mean,std)
    h5Al.close()
    h5DoG.close()
