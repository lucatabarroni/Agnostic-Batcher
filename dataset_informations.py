import numpy as np
import uproot
import os
import pickle

class partitioner():
    def __init__(self,directories,tree_name,tr_size=0.5,te_size=0.3,val_size=0.2):
        self.dir=directories
        self.tn=tree_name
        self.train_size=tr_size
        self.test_size=te_size
        self.validation_size=val_size
        
        self.fl=[]
        self.fs=[]
        self.tot=[]
        
        self.fl_tr=[]
        self.fs_tr=[]

        self.fl_te=[]
        self.fs_te=[]


        self.fl_va=[]
        self.fs_va=[]

        #self.dir_tr=[]
        #self.dir_te=[]
        #self.dir_va=[]
        
        for directory in self.dir:
            list_to_append=[os.path.join(directory, file) for file in os.listdir(directory)]
            self.fl.append(list_to_append)
        
        for i,directory in enumerate(self.fl):
            sizes_to_append=[]
            for file in directory:
                with uproot.open(file) as f:
                    sizes_to_append.append((f[self.tn[i]].num_entries))
            self.fs.append(sizes_to_append)
            
        for ev in self.fs:
            self.tot.append(sum(ev))

        self.set_per_dir=[]
        for ev in self.tot:
            tr=int(self.train_size*ev)
            te=int(self.test_size*ev)
            va=ev-tr-te
            per_dir=[tr,te,va]
            self.set_per_dir.append(per_dir)

        self.perc=[]
        self.total=sum(self.tot)

        for tot_ev_ty in self.tot:
            self.perc.append(tot_ev_ty/self.total)
        
    def train_test_validation_split(self):
        
        for dir_idx,dir in enumerate(self.fl):
            final_file=0
            final_ev=0
            
            train_ev=self.set_per_dir[dir_idx][0]
            tr_ev_fr_dir=0
            complete_tr=False

            test_ev=self.set_per_dir[dir_idx][1]
            te_ev_fr_dir=0
            complete_test=False

            val_ev=self.set_per_dir[dir_idx][2]
            val_ev_fr_dir=0
            complete_val=False
            
            for tr_file_idx in range(final_file,len(dir)):
                file=dir[tr_file_idx]
                for ev_idx in range(final_ev,self.fs[dir_idx][tr_file_idx]):
                    tr_ev_fr_dir+=1
                    if tr_ev_fr_dir==train_ev:
                        complete_tr=True
                        final_file=file
                        self.fl_tr[dir_idx].append(final_file)
                        self.fs_tr[dir_idx].append([final_ev,ev_idx+1])
                        final_ev=ev_idx+1
                        break
                if complete_tr:
                    break
                else:
                    self.fl_tr[dir_idx].append(file)
                    self.fs_tr[dir_idx].append([final_ev,ev_idx+1])
                    final_ev=0
            final_file=tr_file_idx
            
            for te_file_idx in range(final_file,len(dir)):
                file=dir[te_file_idx]
                for ev_idx in range(final_ev,self.fs[dir_idx][te_file_idx]):
                    te_ev_fr_dir+=1
                    if te_ev_fr_dir==test_ev:
                        complete_test=True
                        final_file=file
                        self.fl_te[dir_idx].append(final_file)
                        self.fs_te[dir_idx].append([final_ev,ev_idx+1])
                        final_ev=ev_idx+1
                        break
                if complete_test:
                    break
                else:
                    self.fl_te[dir_idx].append(file)
                    self.fs_te[dir_idx].append([final_ev,ev_idx+1])
                    final_ev=0
            final_file=te_file_idx

            for va_file_idx in range(final_file,len(dir)):
                file=dir[va_file_idx]
                for ev_idx in range(final_ev,self.fs[dir_idx][va_file_idx]):
                    val_ev_fr_dir+=1
                    if val_ev_fr_dir==val_ev:
                        complete_val=True
                        final_file=file
                        self.fl_va[dir_idx].append(final_file)
                        self.fs_va[dir_idx].append([final_ev,ev_idx+1])
                        final_ev=ev_idx+1
                        break
                if complete_val:
                    break
                else:
                    self.fl_va[dir_idx].append(file)
                    self.fs_va[dir_idx].append([final_ev,ev_idx+1])
                    final_ev=0
                    
    def get_train_test_validation(self):
        
        self.fl_tr=[]
        self.fs_tr=[]

        self.fl_te=[]
        self.fs_te=[]
        
        self.fl_va=[]
        self.fs_va=[]

        for i in range(len(self.fl)):
            self.fl_tr.append([])
            self.fs_tr.append([])
            self.fl_te.append([])
            self.fs_te.append([])
            self.fl_va.append([])
            self.fs_va.append([])          

        
        self.train_test_validation_split()
        
        return [self.fl_tr,self.fs_tr],[self.fl_te,self.fs_te],[self.fl_va,self.fs_va]