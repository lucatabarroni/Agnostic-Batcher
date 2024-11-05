import data_Loader
import pickle

dir=['/home/rgw/tabarroni/hepd_data/dataset_4/electrons/',
     '/home/rgw/tabarroni/hepd_data/dataset_4/protons/']
tree_name=['good_ele_tree','int_proton_tree']


object=data_Loader.Loader(dir,tree_name,batch_size=4096,training_size=0.5,test_size=0.3,validation_size=0.2)

with open('train_file_list.pkl','wb') as f:
    pickle.dump(object.train_file_list,f)

with open('train_ev_list.pkl','wb') as f:
    pickle.dump(object.train_ev_list,f)

with open('test_file_list.pkl','wb') as f:
    pickle.dump(object.test_file_list,f)

with open('test_ev_list.pkl' , 'wb') as f:
    pickle.dump(object.test_ev_list,f)

with open('validation_file_list.pkl','wb') as f:
    pickle.dump(object.validation_file_list,f)
        
with open('validation_ev_list.pkl','wb') as f:
    pickle.dump(object.validation_ev_list,f)