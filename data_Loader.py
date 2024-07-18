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
        self.train_file_list=self.train_dataset[0]
        self.train_ev_list=self.train_dataset[1]
        
        self.test_dataset=dataset[1]
        self.test_file_list=self.test_dataset[0]
        self.test_ev_list=self.test_dataset[1]
        
        self.validation_dataset=dataset[2]
        self.validation_file_list=self.validation_dataset[0]
        self.validation_ev_list=self.validation_dataset[1]