from fastai.vision import *

class ImageActiveLearner:
    def __init__(self,path,train_df,get_model,acq_fn=None,label_fn=None,**kwargs):
        """Initialiaze the Learner with the train and test data, and create it with the given params. 
        data_params are the parameters of the TabularDatabunch.
        learn_params are the parameters of the tabular learner other than the Databunch.
        acq_fn is the acq_fn to use. Its signature is :
        acq_fn(learn:Learner,k)
        
        label_fn()
        
        """
        self.path = path
        
        # We store the functions we will use
        self.acq_fn = acq_fn
        self.label_fn = label_fn
        self.get_model = get_model
        
        # Create the Learner
        self.add_learn(train_df,**kwargs)
    
    @classmethod
    def create_databunch(cls,df,valid_pct = 0.2,bs = 32,size = 28,gray = True):
        path = Path("/")

        tfms = get_transforms(do_flip = False)

        data = (ImageList.from_df(df,path)
                .split_by_rand_pct(valid_pct=valid_pct,seed=42)
                .label_from_df()
                .transform(tfms,size = size)
                .databunch(bs = bs)
                .normalize())

        def get_one_channel(batch):
            x,y = batch
            return x[:,0,:,:].unsqueeze(1),y
        get_one_channel._order = 99

        if gray:
            data.add_tfm(get_one_channel)
        data.path = Path()

        return data
        
    def add_learn(self,train_df,**kwargs):

        # Create a Learner 
        path = self.path
        train_data = ImageActiveLearner.create_databunch(df=train_df, **kwargs)
        model = self.get_model()
        learn = Learner(train_data,model,metrics=accuracy)
        
        # Add the Custom Dropout to do MC Dropout
        get_args = lambda dp : {"p" : dp.p}
        convert_layers(learn.model,nn.Dropout,CustomDropout,get_args)
        switch_custom_dropout(learn.model,True)
        
        self.learn = learn
        
    def fit(self,n_epoch,lr):
        """Train the model using one cycle policy and with the training params"""
        n_epoch = listify(n_epoch)
        lr = listify(lr)
        
        for n,l in zip(n_epoch,lr):
            self.learn.fit_one_cycle(n, l)
    
    def inspect(self):
        """Inspect the results of the learning"""
        pass
    
    def acquire(self,acq_df,k,bs=512):
        """Get indexes of the k most interesting labels to acquire from the test set"""
        
        # We add a acquisition Databunch
        acq_data = ImageActiveLearner.create_databunch(df=acq_df, valid_pct=0.,bs=bs)
        idx = self.acq_fn(self.learn,acq_data,k)
        
        return idx
    
    def validate(self,metric,test_df,bs=512,MC_dropout=False):
        """Return a validation score on the test set"""
        learn = self.learn
        
        test_data = ImageActiveLearner.create_databunch(df=test_df, valid_pct=0.,bs=bs)
        
        learn.data = test_data
        
        if MC_dropout:
            preds,y = get_preds_sample(learn,DatasetType.Fix)
            pred = preds.mean(dim=0)
        else:
            pred,y = learn.get_preds(DatasetType.Fix)
        
        score = metric(pred,y)
        return score
        
    def label(self,idx,acq_df):
        """Label the indexes of the acquisition dataset"""
        df_to_label = acq_df.iloc[idx]        
        labeled_df = self.label_fn(df_to_label)
        return labeled_df
    
    @classmethod
    def transfer_rows(cls,train_df,acq_df,idx):
        """Transfer the rows of the acq_df to the train_df"""
        rows = acq_df.iloc[idx]

        train_df = pd.concat([train_df,rows])
        acq_df = acq_df.drop(acq_df.index[idx])
        
        return train_df, acq_df