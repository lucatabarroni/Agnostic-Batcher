from dataset_informations import partitioner
import numpy as np
"""
    The module is devoted to divide the three dataset in batches of given size. 
    It inherits from the module dataset_information and uses its results to define the batches.
    The output are two lists for every set. In the first we will find the name of the files saved in groups of batch-size elements. 
    In the second we find the event ids also saved in groups of batch-size elements.
"""
class batcher(partitioner):
    """
        In the builder as input:
        -the path of the directories, store them in self.dir
        -the names of the tree where to take the elements from the files, store them in self.tn
        -the minimum size of the batches self.batch_size
        -the size of the three different sets, store them in self.train_size, self.test_size and self.validation_size
        -the variable used to decide wether or not to shuffle the dataset, stored in self.shuffle
    """
    def __init__(self, directories, tree_name,batch_size=64,training_size=0.5,testing_size=0.3,validation_size=0.2,shuffle=False):
        # instantiate a list of directory paths dir :list[str]
        self.dir=directories
        # istantiate a list of tree names tn:list[str]
        self.tn=tree_name
        # istantiate the minimum size of the batch
        self.bs=batch_size
        # istantiate the ratio train/dataset
        self.tr_s=training_size
        # istantiate the ratio test/dataset
        self.te_s=testing_size
        # istantiate the ratio validation/dataset
        self.va_s=validation_size
        # istantiate the shuffle of the dataset
        self.shuffle=shuffle

        super().__init__(self.dir,self.tn,self.tr_s,self.te_s,self.va_s,self.shuffle)
        dataset=super().get_train_test_validation()

        # train_set :[list[str].list[int]] file names for train and event ids for train
        self.train_set=dataset[0]
        # test_set :[list[str].list[int]] file names for train and event ids for test
        self.test_set=dataset[1]
        # validation_set :[list[str].list[int]] file names for train and event ids for validation
        self.validation_set=dataset[2]
        
        self.tot_tr=0
        self.tot_te=0
        self.tot_va=0

        # tot_tr :int number of train events
        for dir in self.set_per_dir:
            self.tot_tr+=dir[0]
            self.tot_te+=dir[1]
            self.tot_va+=dir[2]

        self.batch_per_dir=[]

        # batch_per_dir :list[tr_batch,te_batch,va_batch] for each directory number of events in the train,test and validation batches 
        for dir in self.set_per_dir:
            tr_batch=int(np.ceil((dir[0]/self.tot_tr)*self.bs))
            te_batch=int(np.ceil((dir[1]/self.tot_te)*self.bs))
            va_batch=int(np.ceil((dir[2]/self.tot_va)*self.bs))
            self.batch_per_dir.append([tr_batch,te_batch,va_batch])


        #definiamo il numero di batch in base alla directory che ne contiene il numero minore 
        #per ogni set

        # num_batches :list[int] number of batches for each set
        self.num_batches=[-1,-1,-1]
        for set,set_ev in enumerate(self.set_per_dir):
            for set_batch,ev_in_batch in enumerate(self.batch_per_dir[set]):
                if self.num_batches[set_batch]==-1 or self.num_batches[set_batch]>set_ev[set_batch]/ev_in_batch:
                    self.num_batches[set_batch]=int(np.ceil(set_ev[set_batch]/ev_in_batch))

    def train_test_validation_batcher(self):
        """
            train_test_validation_batcher uses the information of the builder to divide the three dataset in batches.
            The output are three couples of lists one for each set. 
            The first list is for the file names and is filled by group of batch-size. The second list is for the id of the events
            and it is also filled by group of batch-size.
        """
            
        self.batches_fl_tr=[]
        self.batches_ev_tr=[]

        # istantiate list of batches
        for i in range(self.num_batches[0]):
            self.batches_fl_tr.append([])
            self.batches_ev_tr.append([])

        
        for i in range(self.num_batches[0]):
            for j in range(len(self.dir)):
                self.batches_fl_tr[i].append(self.train_set[0][j][i*self.batch_per_dir[j][0]:(i+1)*self.batch_per_dir[j][0]])
                self.batches_ev_tr[i].append(self.train_set[1][j][i*self.batch_per_dir[j][0]:(i+1)*self.batch_per_dir[j][0]])

        self.batches_fl_te=[]
        self.batches_ev_te=[]

        for i in range(self.num_batches[1]):
            self.batches_fl_te.append([])
            self.batches_ev_te.append([])

        for i in range(self.num_batches[1]):
            for j in range(len(self.dir)):
                self.batches_fl_te[i].append(self.test_set[0][j][i*self.batch_per_dir[j][1]:(i+1)*self.batch_per_dir[j][1]])
                self.batches_ev_te[i].append(self.test_set[1][j][i*self.batch_per_dir[j][1]:(i+1)*self.batch_per_dir[j][1]])

        self.batches_fl_va=[]
        self.batches_ev_va=[]
            
        for i in range(self.num_batches[2]):
            self.batches_fl_va.append([])
            self.batches_ev_va.append([])

        for i in range(self.num_batches[2]):
            for j in range(len(self.dir)):
                self.batches_fl_va[i].append(self.validation_set[0][j][i*self.batch_per_dir[j][2]:(i+1)*self.batch_per_dir[j][2]])
                self.batches_ev_va[i].append(self.validation_set[1][j][i*self.batch_per_dir[j][2]:(i+1)*self.batch_per_dir[j][2]])

        
        
        return [self.batches_fl_tr,self.batches_ev_tr],[self.batches_fl_te,self.batches_ev_te],[self.batches_fl_va,self.batches_ev_va]
