import numpy as np
import glob
import os
from src import PACKh5raw
from src import PACKDoG
from src import PACKAl
#from src import PACKz
from src import PACKPrepare
print("----Data Manager CLI----")
print("h for help")
def DMhelp():
    print("h : help")
    print("l : list datasets")
    print("q! : force leave")
    print("status : reports status of all datasets")
    print("status DATASETNAME1 DATASETNAME2 ... : reports status of dataset(s)")
    print("process DATASETNAME1 DATASETNAME2 ... : processes dataset(s)")
    print()

processes=["h5raw","h5raw!","DoG","DoG!","Al","Al!","Prepare","Prepare!"]
sortdict=dict(zip(processes,list(range(len(processes)))))
def sortfunc(val):
    return sortdict[val]

def reportstatus(dsetfol):
    print()
    print("Status of dataset:",os.path.split(dsetfol)[-1])
    nothing=True
    indis=[os.path.split(f)[-1] for f in glob.glob(os.path.join(dsetfol,"*")) if os.path.split(f)[-1] in ["h5raw","DoG","Al","Prepare"]]
    for indi in sorted(indis,key=sortfunc):
        print(indi," done")
        nothing=False
    if nothing:
        print("No Preprocessing done")
    print()

def reportallstatus():
    print()
    for dsetfol in glob.glob(os.path.join("data","*")):
        print(os.path.split(dsetfol)[-1],end=": ")
        nothing=True
        indis=[os.path.split(f)[-1] for f in glob.glob(os.path.join(dsetfol,"*")) if os.path.split(f)[-1] in ["h5raw","DoG","Al","Prepare"]]
        for indi in sorted(indis,key=sortfunc):
            print(indi,end=" ")
            nothing=False
        if nothing:
            print("No Preprocessing",end=" ")
        print("done")
    print()

def process(dsetfol,process_commands):
    print("\tApplying",process_commands,"for",dsetfol)
    if process_commands[0]=="all":
        process_commands=["h5raw","DoG","Al","Prepare"]
    existings=[os.path.split(f)[-1] for f in glob.glob(os.path.join(dsetfol,"*"))]
    process_commands_sorted=[]
    skippable=True
    for process_command in sorted(process_commands,key=sortfunc):
        if process_command[-1]=="!":
            process_command=process_command[:-1]
            skippable=False
        process_commands_sorted.append([process_command,skippable])
    print()
    for process_command,skip in process_commands_sorted:
        if  skip and (process_command in existings):
            print("\tSkipping "+process_command)
            print()
            continue
        if process_command=="h5raw":
            print("\tRunning "+process_command)
            PACKh5raw.h5raw(dsetfol)
            print()
        elif process_command=="DoG":
            print("\tRunning "+process_command)
            PACKDoG.DoG(dsetfol)
            print()
        elif process_command=="Al":
            print("\tRunning "+process_command)
            PACKAl.Al(dsetfol)
            print()
        #elif process_command=="z":
        #    print("\tRunning "+process_command)
        #    PACKz.z(dsetfol)
        #    print()
        elif process_command=="Prepare":
            print("\tRunning "+process_command)
            PACKPrepare.Prepare(dsetfol)
            print()
def DMProcessHelp():
    print("Example:")
    print("  Setup process:h5raw DoG Al")
    print("To Apply h5raw, DoG and Al")
    print("----")
    print("h5raw : parse binary to h5file | requirement: original binary file")
    print("DoG : apply DoG | requirement:[h5raw]")
    print("Al : apply alignment | requirement:[DoG]")
    #print("z : apply z alignment | requirement:[Al]")
    print("Prepare : center and rotate using neural network | requirement:[Al]")
    print("----")
    print("all: apply all")
    print("Adding \"!\" to all single commands amke then run again even if already done.")
    print("----")
    print("q: Abort Proccess Setup")
    print()
while True:
    comm=input("DataManager:")
    if comm=="h":
        DMhelp()
        continue
    elif comm=="q!":
        print("Force Leaving")
        break
    elif comm=="l":
        dsetnames=glob.glob(os.path.join("data","*"))
        if len(dsetnames)==0:
            print("--No Datasets--")
        for dsetname in dsetnames:
            print(os.path.split(dsetname)[-1])
        print()
        continue
    elif comm=="status":
        reportallstatus()
        continue
    a=comm.split()
    if len(a)<2:
        print("Invalid command, see help with h")
        continue
    dsetnames=glob.glob(os.path.join("data","*"))
    dsetfiledict={}
    for dsetname in dsetnames:
        dsetfiledict[os.path.split(dsetname)[-1]]=dsetname
    if a[0] not in ["status","process"]:
        print("No command",a[0],"see help with h")
        print()
        continue
    breaking=False
    for n in a[1:]:
        if n not in dsetfiledict:
            print("No dataset named",a[1],"type l to list datasets")
            print()
            breaking=True
            break
    if breaking:
        continue
    if a[0]=="status":
        for dsetname in a[1:]:
            reportstatus(dsetfiledict[dsetname])
    elif a[0]=="process":
        while True:
            print("\t",end="")
            comm=input("Setup process:")
            if comm=="h":
                DMProcessHelp()
                continue
            elif comm=="q":
                break
            else:
                validated=False
                commlist=comm.split()
                if len(commlist)<1:
                    pass
                elif commlist[0]=="all" and len(commlist)==1:
                    validated=True
                elif any([comm not in processes for comm in commlist]):
                    pass
                else:
                    validated=True
            if validated:
                confirm=input("Confirm: Apply "+str(commlist)+" to "+str(a[1:])+" (y/n):")
                if confirm=="y":
                    for dsetname in a[1:]:
                        process(dsetfiledict[dsetname],commlist)
                    break
                else:
                    print("Aborting")
                    break
            else:
                print("Invalid process, see help with h")
                continue
