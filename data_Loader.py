from train_test_validation_batcher import batcher
from keras.utils import Sequence

class Loader(batcher):
    def __init__(self,directories,tree_name,batch_size=64,training_size=0.5,test_size=0.3,validation_size=0.2,shuffle=False):
        self.dir=directories
        self.tn=tree_name
        self.bs=batch_size
        self.tr_s=training_size
        self.te_s=test_size
        self.va_s=validation_size
        self.shuffle=shuffle
        
        super().__init__(self.dir,self.tn,self.bs,self.tr_s,self.te_s,self.va_s,self.shuffle)

        dataset=super().train_test_validation_batcher()
        
        self.train_dataset=dataset[0]
        self.train_file_list=[]
        self.train_ev_list=[]
        for i in range (len(self.train_dataset[0])):
            file_batch=[]
            ev_batch=[]
            for j in range (len(self.train_dataset[0][i])):
                file_batch=file_batch+self.train_dataset[0][i][j]
                ev_batch=ev_batch+self.train_dataset[1][i][j]
            self.train_file_list.append(file_batch)
            self.train_ev_list.append(ev_batch)
        
        self.test_dataset=dataset[1]
        self.test_file_list=[]
        self.test_ev_list=[]
        for i in range (len(self.test_dataset[0])):
            file_batch=[]
            ev_batch=[]
            for j in range (len(self.test_dataset[0][i])):
                file_batch=file_batch+self.test_dataset[0][i][j]
                ev_batch=ev_batch+self.test_dataset[1][i][j]
            self.test_file_list.append(file_batch)
            self.test_ev_list.append(ev_batch)
        
        self.validation_dataset=dataset[2]
        self.validation_file_list=[]
        self.validation_ev_list=[]
        for i in range (len(self.validation_dataset[0])):
            file_batch=[]
            ev_batch=[]
            for j in range (len(self.validation_dataset[0][i])):
                file_batch=file_batch+self.validation_dataset[0][i][j]
                ev_batch=ev_batch+self.validation_dataset[1][i][j]
            self.validation_file_list.append(file_batch)
            self.validation_ev_list.append(ev_batch)
