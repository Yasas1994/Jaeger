import os


class DataPaths:
    
    """Checks files in given paths and returns Validation, Train and Test file paths """
    def __init__(self):
        
        self.validation = [] #validation data paths
        self.train = [] #train data paths 
        self.test = [] #test data paths
        
    def update_data_paths(self,dir_positive, dir_negative, suf_neg=None,suf_pos=None):
        
        self.neg = dir_negative #path to negative class files
        self.pos = dir_positive #path to positive class files
        self.suf_neg = suf_neg # file name suffix 
        self.suf_pos = suf_pos
        
        for path,suf in zip([self.neg,self.pos],[self.suf_neg,self.suf_pos]):
            try:
                for i in os.listdir(path):
                    if suf :
                        if suf in i:
                            self.update(os.path.join(path,i))
                        else:
                            print(f"WARNING! the file {i} does not have the suffix {suf}")
                    else:
                        self.update(os.path.join(path,i))
            except Exception as e:
                print(e)
                
        return self
                    
    def update(self,filename):
        
        if 'test' in filename.lower():
            self.test.append(filename)
        elif 'train' in filename.lower():
            self.train.append(filename)
        elif 'val' in filename.lower():
            self.validation.append(filename)
            
