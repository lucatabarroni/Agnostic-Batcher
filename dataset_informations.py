import numpy as np
import uproot
import os
import pickle
import random

class partitioner():
    def __init__(self,directories,tree_name,tr_size=0.5,te_size=0.3,val_size=0.2,shuffle=False):
        self.dir=directories
        self.tn=tree_name
        self.train_size=tr_size
        self.test_size=te_size
        self.validation_size=val_size
        self.shuffle=shuffle

        self.fl=[]
        for directory in self.dir:
            list_to_append=[os.path.join(directory, file) for file in os.listdir(directory)]
            self.fl.append(list_to_append)

        self.fs=[]
        for i,directory in enumerate(self.fl):
            sizes_to_append=[]
            for file in directory:
                with uproot.open(file) as f:
                    sizes_to_append.append((f[self.tn[i]].num_entries))
            self.fs.append(sizes_to_append)

        self.tot=[]
        for ev in self.fs:
            self.tot.append(sum(ev))

        self.set_per_dir=[]
        for ev in self.tot:
            tr=int(self.train_size*ev)
            te=int(self.test_size*ev)
            va=ev-tr-te
            per_dir=[tr,te,va]
            self.set_per_dir.append(per_dir)

        self.total=sum(self.tot)

        self.perc=[]
        for tot_ev_ty in self.tot:
            self.perc.append(tot_ev_ty/self.total)

        self.tot_list_fl=[]
        self.tot_list_ev=[]
        for i,directory in enumerate (self.fl):
            list_dir_fl=[]
            list_dir_ev=[]
            for j,file in enumerate(directory):
                file_list=[file]*self.fs[i][j]
                ev_list=list(range(self.fs[i][j]))
                list_dir_fl=list_dir_fl+file_list
                list_dir_ev=list_dir_ev+ev_list
            self.tot_list_fl.append(list_dir_fl)
            self.tot_list_ev.append(list_dir_ev)

        self.fl_tr=[]
        self.fs_tr=[]

        self.fl_te=[]
        self.fs_te=[]

        self.fl_va=[]
        self.fs_va=[]

    def get_train_test_validation(self):
        if self.shuffle:
            for i in range(len(self.tot_list_fl)):
                list_1=self.tot_list_fl[i]
                list_2=self.tot_list_ev[i]
                couples=list(zip(list_1,list_2))
                random.shuffle(couples)
                list_1,list_2=zip(*couples)
                self.tot_list_fl[i]=list(list_1)
                self.tot_list_ev[i]=list(list_2)
        
        for i in range(len(self.set_per_dir)):
            tr,te,va=self.set_per_dir[i]
            self.fl_tr.append(self.tot_list_fl[i][:tr])
            self.fs_tr.append(self.tot_list_ev[i][:tr])
            self.fl_te.append(self.tot_list_fl[i][tr:tr+te])
            self.fs_te.append(self.tot_list_ev[i][tr:tr+te])
            self.fl_va.append(self.tot_list_fl[i][tr+te:])
            self.fs_va.append(self.tot_list_ev[i][tr+te:])
        return [self.fl_tr,self.fs_tr],[self.fl_te,self.fs_te],[self.fl_va,self.fs_va]
            
