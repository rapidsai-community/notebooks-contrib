import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tqdm import tqdm

class PlasticcRNN:
    """
    An attentional bi-directional RNN used in the PLASTiCC Challenge
    """
    
    def __init__(self, path, **params):
        self.params = {
            'load_path':path,
            'hidden':64,
            'bottleneck':True,
            'classes':14,
            'num_features':4,
            'embedding_size':4,
            'stratified':True,
            'objective':'multiclassification',
            'metric':'cross_entropy',
            'save_path':'weights',      
            'epochs':100,
            'early_stopping_epochs':10,
            'learning_rate':0.01,
            'batch_size':2048,
            'verbosity':1,
        }
        
        self.params.update(params)
        self._reset()
        
    def _reset(self):
        tf.reset_default_graph()
        self.best_weight = None
        self.loaded_weights = {}
        self._load()
        self.is_training = tf.placeholder(tf.bool)

    def _get_loss(self,labels):
        loss1 = self._get_crossentropy(self.logit,labels)
        loss2 = tf.reduce_mean(tf.pow(self.next_flux - self.next_flue_pred,2))
        loss2 = tf.clip_by_value(loss2*0.1,0,0.1)
        loss3 = self.params.get('lambda',1e-4)*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = loss1+loss2+loss3

    def _get_crossentropy(self,logit,labels):
        NC = self.params['classes']
        with tf.name_scope("Loss"):
            with tf.name_scope("cross_entropy"):
                labels = tf.cast(labels, tf.int32)
                labels = tf.one_hot(labels,NC)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels))
        return loss

    def predict_bottleneck(self,X):
        self.X = X
        self.params['bottleneck'] = True
        self._reset()
        self.logit = self._build()
        count = 0
        yp = []
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self._restore()
            total=X.object_id.unique().shape[0]//self.params.get('batch_size',512)
            for batch in tqdm(self._batch_gen_test(),total=total):
                x,_,epoch = batch
                fdic = {self.inputs:x,self.is_training:0}
                pred = self.sess.run(self.logit,feed_dict=fdic)
                yp.append(pred)
        yp = np.concatenate(yp)
        return yp  

    def _batch_gen_train(self):
        Xt,y = self.X,self.y
        Xt['count'] = Xt['object_id'].map(Xt['object_id'].value_counts())
        count = Xt[['object_id','count']].drop_duplicates(subset=['object_id'])['count'].values
        Xt.drop('count',axis=1,inplace=True)

        B = self.params.get('batch_size',512)
        epochs = 1
        self.col_dic = None
        for epoch in range(epochs):
            s = 0
            for i in range(0,len(count),B):
                ex = min(i+B,len(count))
                e = np.sum(count[i:ex])
                x = self._unstack(Xt.iloc[s:s+e])
                s += e
                yield x,y[i:ex],epoch

    def _batch_gen_test(self):
        Xt = self.X
        Xt['count'] = Xt['object_id'].map(Xt['object_id'].value_counts())
        count = Xt[['object_id','count']].drop_duplicates(subset=['object_id'])['count'].values
        Xt.drop('count',axis=1,inplace=True)
        
        B = self.params.get('batch_size',512)
        epochs = 1
        self.col_dic = None
        for epoch in range(epochs):
            s = 0
            for i in range(0,len(count),B):
                e = min(i+B,len(count))
                e = np.sum(count[i:e])
                x = self._unstack(Xt.iloc[s:s+e])
                s += e
                yield x,None,epoch

    def _unstack(self,df):
        df = df.set_index(['object_id','step']).unstack(-1)
        #print(df.shape)
        #if df.shape[1]%4:
        #    return df
        self._gen_col_dic(df)
        cols = ['flux_delta','flux_err','mjd_delta','passband']
        x = df.values
        step = x.shape[1]//len(set(df.columns.get_level_values(0))) 
        #print(df.shape[1],step)
        x = [np.expand_dims(x[:,self.col_dic[col]:self.col_dic[col]+step],2) for col in cols]
        #print([i.shape for i in x])
        return np.nan_to_num(np.concatenate(x,2))

    def _gen_col_dic(self,df):
        self.col_dic = {}
        for c,col in enumerate(df.columns.get_level_values(0)):
            if col not in self.col_dic:
                self.col_dic[col] = c
  
    def _build(self):
        name = 'RNN'
        
        F = self.params['num_features']
        BT = self.params.get('bottleneck',0) # predict bottleneck
        V = 7 # There are 6 passbands and we add the 7th for the introduced gap
        E = self.params['embedding_size']
        H = self.params['hidden']
        
        
        NC = self.params['classes']
        self.inputs = tf.placeholder(tf.float32,shape=[None,None,F])
        
        with tf.variable_scope(name):
            
            # input is a [B,S,F] tensor
            # B: batch size
            # S: sequence length
            # F: number of features
            # The last feature is passband, which is embedded 
            # and then combined with thre rest features.
            
            net = self.inputs
            self.next_flux = net[:,1:,0] # flux at the next timestamp
            
            net1,net2 = net[:,:,:-1],net[:,:,-1] 
            net2 = self._get_embedding("%s/passband"%(name),net2,V,E)
            net = tf.concat([net1,net2],axis=2)
            
            # Now net is a [B,S,F-1+E] tensor after embedding
            # It is fed into a bidirectional RNN
            
            state = None
            
            cell_name = "GRU"
            args = {"num_units":H}
            net = self._bd_rnn_layer(net,"%s/rnn3"%name,cell_name,args,
                state_fw=state,state_bw=state,output_size=-1,useproject=False)
            
            # Now net is a [B,S,H] tensor 
            # At first layer, the model also learns to predict the flux at the next timestamp
            # as an auxiliary objective function, which warms up model from cold start.
            
            args = {"num_units":H//4,"activation":'relu'}
            self.next_flux_pred = self._bd_rnn_layer(net,"%s/rnn5"%name,cell_name,args,
                state_fw=state,state_bw=state,output_size=1,useproject=True)[:,:-1,0]

            # Feed into a 2nd bidirectional RNN
            args = {"num_units":H//4,"activation":'relu'}
            net = self._bd_rnn_layer(net,"%s/rnn4"%name,cell_name,args,
                state_fw=state,state_bw=state,output_size=-1,useproject=False)
            # Now net is a [B,S,H//4] tensor 
            
            # global-query self-attention
            w = self._get_variable(name, name='attn', shape=[1,args["num_units"]])
            w = tf.expand_dims(w,axis=0)
            atten = tf.nn.softmax(w*net)
            
            # net is [B,S,H//4] and we do a weighted sum along the sequence axis
            # based on global-query attention
            net = tf.reduce_sum(net*atten,axis=1)
            # net is [B,H//4]
            
            if BT == 1:
                # If we are building the network to predict bottleneck features,
                # we want the layer just before the fully-connected (classification)
                # layer. This is our embedded feature space from the RNN.
                return net
            
            # lastly the tensor is fed into a fully connected layer
            # for classification
            net = self._fc(net, NC, layer_name='%s/out'%(name))
            # Now net is a [B,NC] tensor
            return net
    
    def _get_embedding(self, layer_name, inputs, v,m,reuse=False):
        """
            V: vocabulary size
            M: embedding sze
        """ 
        with tf.variable_scope(layer_name.split('/')[-1], reuse=reuse):            
            w = self._get_variable(layer_name, name='w', shape=[v, m])            
            inputs = tf.cast(inputs,tf.int32)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
    
    def _bd_rnn_layer(self,net,name,cell_name,args,
        wf=0.5,wb=0.5,state_fw=None,state_bw=None,
        output_size=1,useproject=True):

        with tf.variable_scope(name.split('/')[-1]):
            cellf = self._get_rnn_cell(cell_name, args)
            cellb = self._get_rnn_cell(cell_name, args)
            if useproject:
                cellb = tf.contrib.rnn.OutputProjectionWrapper(cellb, output_size = output_size)
                cellf = tf.contrib.rnn.OutputProjectionWrapper(cellf, output_size = output_size)
            #net, _ = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32,time_major=False)
            (netf,netw),_ = tf.nn.bidirectional_dynamic_rnn(cellf,cellb,net,dtype=tf.float32,time_major=False,
                initial_state_fw=state_fw,initial_state_bw=state_bw)
            net = netf*wf+netw*wb
            return net
        
    def _get_rnn_cell(self, cell_name, args):
        if cell_name == "BASIC_LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(**args)
        elif cell_name == "GRU":
            cell = tf.contrib.rnn.GRUCell(**args)
        elif cell_name == "LSTM":
            cell = tf.contrib.rnn.LSTMCell(**args)
        elif cell_name == "BLOCK_LSTM":
            cell = tf.contrib.rnn.LSTMBlockCell(**args)
        elif cell_name == "BLOCK_GRU":
            cell = tf.contrib.rnn.GRUBlockCell(**args)
        elif cell_name == "NAS":
            cell = tf.contrib.rnn.NASCell(**args)
        else:
            print("Unknown cell name", cell_name)
            assert 0

        return cell
    
    def _get_variable(self, layer_name, name, shape):
        if len(shape)>1:
            return self._get_weight_variable(layer_name, name, shape)
        else:
            return self._get_bias_variable(layer_name, name, shape)
    
    def _get_weight_variable(self, layer_name, name, shape, L2=1):
        wname = '%s/%s:0'%(layer_name,name)

        if self.weights is None or wname not in self.weights:
            w1 =  tf.get_variable(name,initializer=tf.contrib.layers.xavier_initializer(),
                 shape = shape)
            print('{:>23} {:>23}'.format(wname, 'randomly initialize'))
        else:
            w1 = tf.get_variable(name, shape = shape,
                initializer=tf.constant_initializer(value=self.weights[wname],dtype=tf.float32))
            self.loaded_weights[wname]=1
        if wname != w1.name:
            print('Variable name mismatch')
            print(wname,w1.name)
            assert False
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w1)*L2)
        return w1
    
    
    def _get_bias_variable(self, layer_name, name, shape, L2=1):
        bname = '%s/%s:0'%(layer_name,name)

        if self.weights is None or bname not in self.weights:
            b1 = tf.get_variable(name,shape=shape,initializer=tf.constant_initializer(0))
            print('{:>23} {:>23}'.format(bname, 'randomly initialize'))
        else:
            b1 = tf.get_variable(name,shape=shape,initializer=tf.constant_initializer(value=self.weights[bname],dtype=tf.float32))
            self.loaded_weights[bname]=1
        if bname != b1.name:
            print(bname,b1.name)
            assert False
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b1)*L2)
        return b1
    
    def _fc(self, x, fan_out, layer_name, fan_in=None, activation=None, L2=1, use_bias=True):
        if fan_in is None:
            fan_in=x.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name.split('/')[-1]):
            w,b = self._get_fc_weights(fan_in, fan_out, layer_name)
            net = tf.matmul(x,w)
            if use_bias:                
                net = tf.nn.bias_add(net, b)
            net = self._activate(net, activation)
        return net
    
    def _get_fc_weights(self, fan_in, fan_out, layer_name):
        w1 = self._get_variable(layer_name, name='weights', shape=[fan_in,fan_out])
        b1 = self._get_variable(layer_name, name='bias', shape=[fan_out])
        return w1,b1
    
    def _activate(self, net, activation):
        if activation=="relu":
            net = tf.nn.relu(net)
        elif activation == 'leaky':
            net = self._leaky(net, alpha = 0.1)
        elif activation == "sigmoid":
            net = tf.nn.sigmoid(net)        
        elif activation == "softmax":            
            net = tf.nn.softmax(net)        
        elif activation == "elu":
            net = tf.nn.elu(net)
        elif activation == "tanh":
            net = tf.nn.tanh(net)
        return net
    
    def _load(self):
        load_path = self.params.get('load_path',None)
        if self.best_weight is not None:
            load_path = self.best_weight
        if load_path is not None:
            self.weights = np.load(load_path, allow_pickle=True).item()
        else:
            self.weights = None

    def _restore(self,only_once=True):
        var_list = tf.trainable_variables()
        for var in var_list:
            if self.weights and var.name in self.weights:
                if only_once and self.loaded_weights and var.name in self.loaded_weights:
                    continue

                assign_op = var.assign(self.weights[var.name])
                self.sess.run(assign_op)
                self.loaded_weights[var.name] = 1
                #if only_once:
                print("restore %s"%var.name)
