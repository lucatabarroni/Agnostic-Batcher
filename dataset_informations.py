import numpy as np
import uproot
import os
import pickle
import random
"""
    The module is devoted to divide the whole dataset in the three sets. Each dataset needs to respect the proportion of the different directory's populations.
    As input it takes the path of the directories and in output it gives six couples of lists, two for each set.
    In the first list of the couple we store the names of the files that will form the set. In the second list we save the event_id of the events inside the files that will form the set.
"""
class partitioner():
    """
        In the builder as input:
        -the path of the directories, store them in self.dir
        -the names of the tree where to take the elements from the files, store them in self.tn
        -the size of the three different sets, store them in self.train_size, self.test_size and self.validation_size
        -the variable used to decide wether or not to shuffle the dataset, stored in self.shuffle
    """
    def __init__(self,directories,tree_name,tr_size=0.5,te_size=0.3,val_size=0.2,shuffle=False):
        # instantiate a list of directory paths dir :list[str]
        self.dir=directories
        # instantiate a list of tree names tn:list[str]
        self.tn=tree_name
        # instantiate the ratio train/dataset
        self.train_size=tr_size
        # instantiate the ratio test/dataset
        self.test_size=te_size
        # instantiate the ration validation/dataset
        self.validation_size=val_size
        # istantiate the shuffle of the dataset
        self.shuffle=shuffle

        # fl :list[list[str]] save file names for each directory 
        self.fl=[]
        for directory in self.dir:
            list_to_append=[os.path.join(directory, file) for file in os.listdir(directory)]
            self.fl.append(list_to_append)

        # fs :list[list[int]] save the number of events of each file 
        self.fs=[]
        for i,directory in enumerate(self.fl):
            sizes_to_append=[]
            for file in directory:
                with uproot.open(file) as f:
                    sizes_to_append.append((f[self.tn[i]].num_entries))
            self.fs.append(sizes_to_append)

        # tot :list[int] save the number of events in each directory
        self.tot=[]
        for ev in self.fs:
            self.tot.append(sum(ev))

        # set_per_dir :list[list[int]] save [train,test,validation] events for each directory
        self.set_per_dir=[]
        for ev in self.tot:
            tr=int(self.train_size*ev)
            te=int(self.test_size*ev)
            va=ev-tr-te
            per_dir=[tr,te,va]
            self.set_per_dir.append(per_dir)

        # save the total number of events
        self.total=sum(self.tot)

        # perc :list[float] save the percentage of events in each directory
        self.perc=[]
        for tot_ev_ty in self.tot:
            self.perc.append(tot_ev_ty/self.total)

        # tot_list_fl :list[list[str]] save the names of the files in the directory repeated as many times as the number of events in each file
        # tot_list_ev :list[list[int]] save the list of events id of each file in each directory
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
        """ The method get_train_test_validation returns three couple of lists. Each couple refers to one set.
            The first element of the first couple is self.fl_tr that is a list of lists. It has as many elements as 
            the number of directories and it contains the names of the files that go in the train set. On the other hand 
            the second element of the first couple contains self.fs_tr. If in self.fl_tr the elements are the file names
            that go in the train set, in self.fs_tr we store the event_id that go in the train set.
        """
        # shuffle tot_list_fl and tot_list_ev
        if self.shuffle:
            for i in range(len(self.tot_list_fl)):
                list_1=self.tot_list_fl[i]
                list_2=self.tot_list_ev[i]
                couples=list(zip(list_1,list_2))
                random.shuffle(couples)
                list_1,list_2=zip(*couples)
                self.tot_list_fl[i]=list(list_1)
                self.tot_list_ev[i]=list(list_2)

        # fl_tr :list[str] save the file names for train
        # fs_tr :list[int] id of the events for train
        for i in range(len(self.set_per_dir)):
            tr,te,va=self.set_per_dir[i]
            self.fl_tr.append(self.tot_list_fl[i][:tr])
            self.fs_tr.append(self.tot_list_ev[i][:tr])
            self.fl_te.append(self.tot_list_fl[i][tr:tr+te])
            self.fs_te.append(self.tot_list_ev[i][tr:tr+te])
            self.fl_va.append(self.tot_list_fl[i][tr+te:])
            self.fs_va.append(self.tot_list_ev[i][tr+te:])
        return [self.fl_tr,self.fs_tr],[self.fl_te,self.fs_te],[self.fl_va,self.fs_va]
            
