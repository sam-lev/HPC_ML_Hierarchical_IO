#!/home/sci/samlev/anaconda3/envs/tf2/bin/python3.5
#SBATCH --time=41:06:66 # walltime, abbreviated by -t
#SBATCH --mem=110G
#SBATCH --job-name="shmerp"
#SBATCH -o model_shallow.out-%j # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e model_shallow.err-%j # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

import argparse
import os
import sys

#import sklearn.metrics
import io

# Personal Data Flow
sys.path.append(os.getcwd())

#import tensorflow as tf                                                                                                            
import tensorflow as tf #.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.disable_eager_execution()

import LyCNN.datapack as datapack

from tensorpack import *
from tensorpack.models import BatchNorm
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils import logger
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.train import launch_train_with_config
#from tensorpack.tfutils.symbolic_functions import prediction_incorrect

#from tensorpack.dataflow import LocallyShuffleData

#from tensorpack.dataflow import *
import multiprocessing as mp
import copy

from PIL import Image

import numpy as np

""" slurm gpu test 
"""
def get_available_gpus():
   from tensorflow.python.client import device_lib
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']



"""
Lymphoma Deep Convolutional Neural Net Based on http://arxiv.org/abs/1608.06993
Code similar to Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
See also: https://github.com/YixuanLi/densenet-tensorflow/blob/7793c03a9c9dc009e4151aa4b7a74f0e62583973/cifar10-densenet.py
Dataflow formated similar to cifar10 http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/dataflow/dataset/cifar.html 
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
 @article{Huang2016Densely,
 		author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
 		title = {Densely Connected Convolutional Networks},
 		journal = {arXiv preprint arXiv:1608.06993},
 		year = {2016}
 }
Call Example:  
>> Train listing GPU to use and basic training param:
python3 denseNet_Lymphoma.py --gpu 0,1 (--num_gpu 4) (--load model-xxx) --drop_1 100 --drop_2 200 --depth 40 --max_epoch 368 --tot train
>> Train Using Desired number of GPU and loading model partially trained:
python3 denseNet_Lymphoma.py --num_gpu 8 --tot train --model_name 8_gpu_d1_58_d2_98_phase_3 --lr_0 0.01 --drop_1 58 --lr_1 0.001 --drop_2 98 --lr_2 0.0001 --max_epoch 134 --gpu_frac 0.99 --batch_size 2 --load train_log/4_gpu_d1_60_d2_160_d3_220_max_256-first63-second113-max149/model-398208
>> Predict Class of files in folder '../data/Unknon/' + unknown_dir 
>> and write prediction to '../data/Unknown/'+unknown_dir+'/predictions/'
python3 denseNet_Lymphoma.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load train_log/8_gpu_d1_58_d2_98_phase_3-first58-second98-max134/model-523136 --unknown_dir A
"""

class Model(ModelDesc):
   def __init__(self, depth, image_size, lr_init, kernels, kernel_size, expansion, class_0, class_1, drop_rate, drop_pattern, bn_momentum, skip_norm, train_or_test):
      super(Model, self).__init__()
      self.step = tf.train.get_or_create_global_step()
      self.N = int((depth - 4)  / 3)
      self.image_size = image_size
      
      self.growthRate = expansion 
      self.filters_init = kernels
      
      self.lr_init = tf.get_variable("lr_init", initializer = lr_init)
      
      graph = None
      self.drop_rate = None
      self.drop_pattern = None
      self.train_or_test= train_or_test
      self.class_0 = None
      self.class_1 = None
      print("Cardinality class_0: ", class_0)
      print("Cardinality class_0: ", class_1)
      self.skip_norm = skip_norm if skip_norm is not None else depth+1
      self.bn_momentum = bn_momentum
      
      print("traing or test:" , self.train_or_test)
      print("drop ", self.drop_rate)
      
      if train_or_test==True:
         with tf.variable_scope('test_param') as scope:
            dr = 1.0-drop_rate
            self.drop_rate = 1.0-drop_rate #tf.get_variable("drop_rate", initializer=dr)#, dtype=tf.float32)        
            self.train_or_test= True#tf.get_variable("is_train",initializer=train_or_test)
            if class_0 == class_1:
               self.class_0 = 1.0#tf.get_variable("class_0",initializer=1.0)#, dtype=tf.float32) 
               self.class_1 = 1.0#tf.get_variable("class_1", initializer=1.0)#, dtype=tf.float32) 
            else:
               self.class_0 = class_0#tf.get_variable("class_0", initializer=class_0)#, dtype=tf.float32) 
               self.class_1 = class_1#tf.get_variable("class_1",initializer=class_1)#, dtype=tf.float32) 
         self.drop_pattern = int(drop_pattern)
      else:                                                                           
         self.drop_rate = 1.0#tf.get_variable('drop_rate', initializer=1.0)                                  
         self.train_or_test = False #tf.get_variable('is_training', initializer=False)                         
         self.class_0 = 1.0#tf.get_variable('class_0', initializer=1.0)                                                            
         self.class_1 = 1.0#tf.get_variable('class_1', initializer=1.0)
         self.drop_pattern = 0#self.N*3
         
      self.kernel_size = kernel_size 
      
      self.weight_decay_rate = 1e-4
   # depricated
   def _get_inputs(self):
      return [InputDesc(tf.float32, [None, self.image_size, self.image_size, 3], 'input'),
              InputDesc(tf.int32, [None], 'label') 
      ]
   # non-depricated
   def inputs(self):
      return [tf.TensorSpec((None, self.image_size, self.image_size, 3), tf.float32, 'input'),tf.TensorSpec((None,), tf.int32, 'label')]

   def _build_graph(self, input_vars):
      image, label = input_vars
      image = tf.image.convert_image_dtype(image, dtype = tf.float32)
      ctx = get_current_tower_context()
      print("train or test tower context?", ctx.is_training)
      
      tf.summary.image("Input Image", image[0:20], max_outputs=20)#.astype("uint8"))

      def conv(name, l, channel, stride, kernel_size):
         #rand_seed = np.random.randint(2**32-1)
         #np.random.seed(None)
         conv2d_he = Conv2D(name, l, channel, kernel_size, stride=stride,nl=tf.identity, use_bias=False,
                            W_init = tf.variance_scaling_initializer(dtype=tf.float32))
         #tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))#tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False))
         #np.random.seed(rand_seed)
         return conv2d_he
      
      def batch_norm(scope, name, layer, decay, layer_num, norm_pattern, training):
         with tf.variable_scope(scope) as s:
            if training:
               layer = BatchNorm(name, layer) if layer_num%norm_pattern != 0 else layer
            else:
               if decay is not None:
                  layer = BatchNorm(name, layer, decay=decay, use_local_stat=False) if layer_num%norm_pattern != 0 else layer
               else:
                  layer = BatchNorm(name, layer, use_local_stat=False) if layer_num%norm_pattern != 0 else layer
         return layer
      
      def add_layer(name, l, kernel_size, growth_rate, drop_rate, training, layer_num, drop_pattern, bn_momentum, skip_norm):
         shape = l.get_shape().as_list()
         in_channel = shape[3] 
         with tf.variable_scope(name) as scope:
            # layer num mod 1 for bnorm every layer
            c = batch_norm(name, 'bn.{}'.format(layer_num), l, bn_momentum, layer_num, skip_norm, training) #epsilon=0.001
            c = tf.nn.relu(c)
            c = conv('conv1', c, growth_rate, 1, kernel_size)
            l = tf.concat([c, l], 3)
            if drop_pattern!=0 and layer_num%drop_pattern == 0:
               spatial_drop = tf.shape(l)
               # drop every layer mod drop_pattern, drop_pattern == 0 if no drop wanted
               l = tf.cond(tf.equal(tf.constant(training), tf.constant(True)), lambda: tf.nn.dropout(l, keep_prob = tf.constant(drop_rate),noise_shape=[spatial_drop[0], 1, 1, spatial_drop[3]], name='dropblock'), lambda: l)
         return l
      
      def add_transition(name, l, drop_rate, training, drop_pattern, transition_number, bn_momentum):
         shape = l.get_shape().as_list()
         in_channel = shape[3]
         with tf.variable_scope(name) as scope:
            l = batch_norm(name, 'bntransit.{}'.format(transition_number), l, bn_momentum, 42, 43, training)
            l = tf.nn.relu(l)
            l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
            if drop_pattern!=0:
               l = tf.cond(tf.equal(tf.constant(training), tf.constant(True)), lambda: tf.nn.dropout(l, keep_prob = tf.constant(drop_rate), name='droptransition'), lambda: l)
            l = AvgPooling('pool', l, 2)
         return l
      
      def dense_net(name):
         
         l = conv('conv0',image, self.filters_init , 1, self.kernel_size)
         
         with tf.variable_scope('block1') as scope:
            for i in range(self.N):
               #(name, l, kernel_size, growth_rate, drop_rate, training, layer_num, drop_pattern):
               l = add_layer(name='dense_layer.{}'.format(i), l=l, kernel_size=self.kernel_size,
                             growth_rate=self.growthRate, drop_rate=self.drop_rate,
                             training=self.train_or_test, layer_num=i, drop_pattern=0,
                             bn_momentum=self.bn_momentum, skip_norm=self.skip_norm)
               
            l = add_transition(name='transition1', l=l , drop_rate=self.drop_rate,
                               training=self.train_or_test, drop_pattern=self.drop_pattern, transition_number=1,
                               bn_momentum=self.bn_momentum)
            
         with tf.variable_scope('block2') as scope:
            for i in range(self.N):
               l = add_layer('dense_layer.{}'.format(i), l, self.kernel_size, self.growthRate,
                             self.drop_rate, self.train_or_test, i, self.drop_pattern,
                             self.bn_momentum, self.skip_norm)
            l = add_transition('transition2', l, self.drop_rate, self.train_or_test, self.drop_pattern, 2, self.bn_momentum)
         
         with tf.variable_scope('block3') as scope:
            for i in range(self.N):
               l = add_layer('dense_layer.{}'.format(i), l, self.kernel_size, self.growthRate,
                             self.drop_rate, self.train_or_test, i,  self.drop_pattern, self.bn_momentum, self.skip_norm)
         
         l = batch_norm(name, 'bnlast', l, self.bn_momentum, 42, 42+1, self.train_or_test)
         l = tf.nn.relu(l)
         l = GlobalAvgPooling('gap', l)
         logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
         
         return logits
      
      def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
         #with tf.name_scope('prediction_incorrect'):
         x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
         return tf.cast(x, tf.float32, name=name)
      
      logits = dense_net("dense_net") #map probabilities to real domain
      
      prob = tf.nn.softmax(logits, name='output')  #a generalization of the logistic function that "squashes" a K-dim vector z  of arbitrary real values to a K-dim vector sigma( z ) of real values in the range [0, 1] that add up to 1.

      factorbl = (self.class_0+self.class_1)/(2*self.class_0)#tf.divide(tf.add(self.class_0, self.class_1), tf.multiply(tf.constant(2.0,dtype=tf.float32), self.class_0))
      factordl = (self.class_0+self.class_1)/(2*self.class_1)#tf.divide(tf.add(self.class_0, self.class_1), tf.multiply(tf.constant(2.0,dtype=tf.float32), self.class_1))
      class_weights = tf.constant([factorbl, factordl])
      weights = tf.gather(class_weights, label)
      
      cost = tf.losses.sparse_softmax_cross_entropy(label, logits, weights=weights) #False positive 3* False negatives so adjust weight by factor
      cost = tf.reduce_mean(cost, name='cross_entropy_loss') #normalize
      
      wrong = prediction_incorrect(logits, label)
      
      # monitor training error
      add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
      
      # weight decay on all W
      wd_reg = tf.constant(self.weight_decay_rate, dtype=tf.float32)
      wd_cost = tf.multiply(wd_reg, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
      add_moving_summary(cost, wd_cost)
      
      add_param_summary(('.*/W', ['histogram']))   # monitor W

      self.cost = tf.add_n([cost, wd_cost], name='cost')
      return self.cost
   

   def build_graph(self, image, label):
      return self._build_graph((image,label))
   nondepreicated_reference = """image = image
   label = label
   #image = image / 128.0 - 1
   def conv(name, l, channel, stride):
   rand_seed = np.random.randint(2**32-1)
   np.random.seed(101)
   conv2d_xav = Conv2D(name, l, channel, 6, stride=stride,
   nl=tf.identity, use_bias=False,
   W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False)) #factor=np.sqrt(2.0/(6+channel)), mode='FAN_IN', uniform=False, seed=None, dty
   np.random.seed(rand_seed)
   return conv2d_xav
   
   def add_layer(name, l):
   shape = l.get_shape().as_list()
   in_channel = shape[3]
   with tf.variable_scope(name) as scope:
   c = BatchNorm('bn1', l)
   c = tf.nn.relu(c)
   c = conv('conv1', c, self.growthRate, 1)
   l = tf.concat([c, l], 3)
   return l
   
   def add_transition(name, l, drop_rate, training):
   shape = l.get_shape().as_list()
   in_channel = shape[3]
   with tf.variable_scope(name) as scope:
   l = BatchNorm('bn1', l)
   l = tf.nn.relu(l)
   l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
   l = AvgPooling('pool', l, 2)
   #drop_out = tf.nn.dropout(l, keep_prob)
   
   l = tf.layers.dropout(l, rate = drop_rate, training=training)
   return l
   
   def dense_net(name):
   l = conv('conv0',image,  self.filters_init , 1)
   with tf.variable_scope('block1') as scope:
   
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = add_transition('transition1', l ,self.drop_rate, self.train_or_test)
   
   with tf.variable_scope('block2') as scope:
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = add_transition('transition2', l, self.drop_rate, self.train_or_test)
   
   with tf.variable_scope('block3') as scope:
   
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = BatchNorm('bnlast', l)
   l = tf.nn.relu(l)
   l = GlobalAvgPooling('gap', l)
   logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
   
   return logits
   
   def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
   with tf.name_scope('prediction_incorrect'):
   x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
   return tf.cast(x, tf.float32, name=name)
   
   logits = dense_net("dense_net") #map probabilities to real domain                                                  
   prob = tf.nn.softmax(logits, name='output')  #a generalization of the logistic function that "squashes" 
   factorbl = (self.class_0+self.class_1)/(2.0*self.class_0)
   factordl = (self.class_0+self.class_1)/(2.0*self.class_1)
   class_weights = tf.constant([factorbl, factordl])#factor,(1-factor)])#factor, 1.0-factor]) #dl 730 bl 1576          
   weights = tf.gather(class_weights, label)
   cost = tf.losses.sparse_softmax_cross_entropy(label, logits, weights=weights) #False posit                          
   cost = tf.reduce_mean(cost, name='cross_entropy_loss') #normalize                                                  
   wrong = prediction_incorrect(logits, label)
   # monitor training error                                                                                               
   add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
   # weight decay on all W                                                    
   wd_cost = tf.multiply(1e-3, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
   add_moving_summary(cost, wd_cost)
   add_param_summary(('.*/W', ['histogram']))   # monitor W                              
   self.cost = tf.add_n([cost, wd_cost], name='cost')
   return self.cost"""
   
   def _get_optimizer(self):
      lr = tf.get_variable('learning_rate', initializer=self.lr_init, trainable=False)
      tf.summary.scalar('learning_rate', lr)
      init_base =  tf.multiply(tf.constant(-1.0), tf.divide(tf.log(self.lr_init),tf.log(tf.constant(10.0))))
      momentum = tf.cond( tf.multiply(tf.constant(-1.0), tf.divide(tf.log(lr),tf.log(tf.constant(10.0)))) > init_base,
                          lambda: tf.constant(0.9), lambda: tf.constant(0.88))
      momentum = tf.cond( tf.multiply(tf.constant(-1.0),tf.divide(tf.log(lr),tf.log(tf.constant(10.0)))) > tf.add(tf.constant(1.0), init_base),
                          lambda: tf.constant(0.95), lambda: momentum)
      momentum = tf.cond( tf.multiply(tf.constant(-1.0),tf.divide(tf.log(lr),tf.log(tf.constant(10.0)))) > tf.add(tf.constant(2.0), init_base),
                          lambda: tf.constant(0.99), lambda: momentum)
      momentum = tf.cond( tf.equal(tf.constant(self.train_or_test), tf.constant(False)), lambda: tf.constant(0.99), lambda: momentum)
      return tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)#return tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999,epsilon=1e-08)
   
   def optimizer(self):
      return self._get_optimizer()

def get_data(train_or_test, shuffle = None, image_size = None, scale_size = None
             , scale = None, multi_crop = None, crop_per_case = None, normalize = None
             , unknown_dir = None, original_dir=None):
   isTrain = train_or_test == 'train'
   isVal = train_or_test == 'val'
   #ds = FakeData([[args.batch_size*10, 224, 224, 3], [args.batch_size*10]], 1000, random=False, dtype='uint32')

   if args.data_dir is None:
      dir = '../data'
   else:
      dir = args.data_dir

   if args.model_name is None:
      args.model_name = "no_name"
   mem_log = 'train_log/'+args.model_name+'/memory_log.txt'
   if not os.path.exists(mem_log):
      os.mkdir(mem_log)
   if isTrain or isVal:
      print("    >>>> Using lymphoma 2 dataflow with tensorfpack")
      ds = datapack.lymphoma2(train_or_test, image_size=image_size, scale_size=scale_size
                              , scale=scale, multi_crop=multi_crop, crop_per_case=crop_per_case
                              , normalize=normalize, shuffle=shuffle, dir=dir
                              , unknown_dir=unknown_dir, original_dir=original_dir, idx=None)
   else:
      print( "   >>>> Using IDX conversions for training, allows for resolution dependence ")
      ds = datapack.lymphoma2ZIDX(train_or_test, image_size=image_size, scale_size=scale_size
                              , scale=scale, multi_crop=multi_crop, crop_per_case=crop_per_case
                              , normalize=normalize, shuffle=shuffle, dir=dir
                              , unknown_dir=unknown_dir, original_dir=original_dir
                              ,resolution=args.frac_res, idx=True, mode='r', memory_profile=mem_log)
   args.unique_samples = ds.unique_samples
   
   if train_or_test == 'train':
      args.class_0 = ds.class_0 
      args.class_1 = ds.class_1
   else:
      args.class_0 = 1.0
      args.class_1 = 1.0
   if args.class_weights:
      args.class_0 = args.class_weights[0]
      args.class_1 = args.class_weights[1]
   
   aug_map_func = lambda dp: [dp[0], dp[1]]
   augmentors = []
   augmentor = None
   if isTrain:
      if bool(args.aug_he):
         augmentors = [
            #and dividing by the standard deviation
            datapack.HematoEAug((0.3, 1.4, np.random.randint(2**32-1), True, 1.0)),
            #datapack.NormStainAug(True),
            imgaug.Flip(horiz=True),
         ]
      elif bool(args.aug_randnorm) or bool(args.aug_randhe):
         prob_norm = args.aug_randnorm if int(args.aug_randnorm) != 0 else 0
         prob_he = args.aug_randhe if int(args.aug_randhe) != 0 else 0
         if prob_norm!=0 and prob_he!=0:
            print(" >>>>>>> augmenting with normalization, H&E permutation, and flip.")
            #p_aug = np.random.RandomState(None).uniform(low=0, high=1.0, size=1)[0]
            #if p_aug <= 0.5:
            augmentors = [
               datapack.HematoEAug((0.4, 1.4, np.random.randint(2**32-1), True, prob_he)),
               datapack.NormStainAug((True, prob_norm)),
               imgaug.Flip(horiz=True),
            ]
            #else:
            #   augmentors = [
            #      #datapack.HematoEAug((0.4, 1.4, np.random.randint(2**32-1), True, prob_he)),
            #      datapack.NormStainAug((True, prob_norm)),
	    #      imgaug.Flip(horiz=True),
            #   ]
         elif prob_norm!=0 and prob_he==0:
            print(">>>>>>>> Augmenting with nomalization and flip.")
            augmentors = [
               datapack.NormStainAug((True, prob_norm)),
               #datapack.HematoEAug((0.4, 1.7, np.random.randint(2**32-1), True, prob_he)),
               imgaug.Flip(horiz=True),
            ]
         else:
            print(">>>>>>>> Augmenting with H&E permutation and flip.")
            augmentors = [
               #datapack.NormStainAug((True, prob_norm)),
               datapack.HematoEAug((0.4, 1.4, np.random.randint(2**32-1), True, prob_he)),
               imgaug.Flip(horiz=True),
               imgaug.Brightness(delta=10.0, clip=True),
               imgaug.Contrast(factor_range=(0.7,1.3),clip=True),
            ]
      else:
         augmentors = [
            imgaug.Flip(horiz=True)
            ]
      augmentor = imgaug.AugmentorList(augmentors)
      aug_map_func=lambda dp: [augmentor.augment(dp[0].astype("uint8")),dp[1]]
   else:
      prob_he = args.aug_randhe if args.aug_randhe != 0. else 0
      if bool(args.aug_norm):
         augmentors = [
            #datapack.HematoEAug((0.7, 1.3, np.random.randint(2**32-1), True)),
            datapack.NormStainAug((True,1.0)),
         ]
      elif prob_he != 0:
         print(">>>>>>>> Augmenting with H&E permutation and flip.")
         augmentors = [
            #datapack.NormStainAug((True, prob_norm)),                                                                           
            datapack.HematoEAug((0.4, 1.6, np.random.randint(2**32-1), True, prob_he)),
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(delta=20.0, clip=True),                                                                           
            imgaug.Contrast(factor_range=(0.5,1.5),clip=True),
         ]
      else:
         augmentors = []
         
   print(">>>>>>> Data Set Size: ", ds.size())
   
   batch_size = None
   
   if train_or_test == 'test':
      print(">> Testing ", ds.size(), " images.")
      batch_size = ds.size()
   else:
      batch_size = args.batch_size
   
   if isTrain:
      print("multiprocess/thread mp = ", args.mp)
      if args.mp==0:
         print(">>>>>>>>>>>>>     non-multiprocess")
         ds = AugmentImageComponent(ds, augmentors, copy=True)
         ds = BatchData(ds, batch_size, remainder=not isTrain)
         #ds = PrefetchData(ds,  nr_prefetch=batch_size*4, nr_proc=1)
      elif args.mp==1:
         print(">>>>>>>>>>>>>     multiprocess, no augmented process mapping mp = 1")
         ds = AugmentImageComponent(ds, augmentors, copy=True) 
         ds = PrefetchData(ds,  nr_prefetch=batch_size*4, nr_proc=mp.cpu_count()//4)
         ds = BatchData(ds, batch_size, remainder=not isTrain)
         print("       >>>>>>    using ",mp.cpu_count()," processes.")
      elif args.mp==2:
         print(">>>>>>>>>>>>>     multiprocess mapping mp = ", args.mp)
         print("    >>>>>>>>>> Creating multiprocess with ", mp.cpu_count(), " total processes.")
         ds = MultiProcessMapData(ds,
                                  map_func=aug_map_func,
                                  nr_proc=4)#mp.cpu_count()//4)
         ds = PrefetchDataZMQ(ds, nr_proc = 1)
         ds = BatchData(ds, batch_size, remainder=not isTrain)
      else:
         print(">>>>>>>>>>>>>     multithread mp=",args.mp)
         ## Multithreaded
         ds = MultiThreadMapData(ds,
                                 nr_thread= mp.cpu_count()//16,# if args.num_gpu > 1 else 2,
                                 map_func = aug_map_func,
                                 buffer_size=batch_size*4)# if args.num_gpu > 1 else batch_size*2)
         ds = PrefetchDataZMQ(ds, nr_proc = 1)
         
         ds = BatchData(ds, batch_size, remainder=not isTrain)
      print(">>> Done. |Data|/Batch Size = ", ds.size())
      return ds
   else:
      if len(augmentors) != 0:
         ds = AugmentImageComponent(ds, augmentors, copy=True)
      return BatchData(ds, batch_size, remainder=not isTrain)

def get_config(train_or_test, train_config = None, load_model = None):
   isTrain = train_or_test == 'train'
   if args.model_name is None:
      args.model_name = "no_name"
   log_dir = 'train_log/'+args.model_name
   logger.set_logger_dir(log_dir, 'n')
   
   dataset_train = 1
   dataset_val = None
   steps_per_epoch = 0
   
   # prepare dataset
   # dataflow structure [im, label] in parralel
   if isTrain:
      print(">>>>>> Loading training and validation sets")
      dataset_train = get_data('train', image_size = args.image_size, scale_size = args.scale_size,  scale = args.scale, multi_crop=args.multi_crop, crop_per_case = args.crop_per_case, normalize = args.aug_norm, shuffle = True)
      
      steps_per_epoch = dataset_train.size()#/args.num_gpu if (args.mp != 0 or args.mp != 1)  else dataset_train.size()# = |data|/(batch size * num gpu)
      
      dataset_val = get_data('val', image_size = args.image_size, scale_size = args.scale_size, scale = args.scale, multi_crop=args.multi_crop, crop_per_case = args.crop_per_case, normalize = args.aug_norm, shuffle = False)
      
   drop_rate = args.drop_out if args.drop_out is not None else 0.0
   
   print(" >>>>>>>>>> Steps Per Epoch: ", steps_per_epoch)
   print(">>>>>> Constructing Neural Network...")
   
   denseModel = Model(depth=args.depth, image_size = args.scale_size, lr_init = args.lr_init, kernels = args.kernels, kernel_size = args.kernel_size, expansion=args.expansion, class_0 = args.class_0, class_1 = args.class_1, drop_rate = drop_rate, drop_pattern = args.drop_pattern, bn_momentum = args.bn_momentum, skip_norm = args.skip_norm, train_or_test=isTrain)
   
   if isTrain:
      print("Setting up training configuration: callbacks, validation checks and hyperparameter scheduling.")
      return TrainConfig(
         dataflow=dataset_train,
         callbacks=[
            MovingAverageSummary(),
            ModelSaver(), # Record state graph at intervals during epochs
            InferenceRunner(input=dataset_val,
                            infs=[ScalarStats('cost'), ClassificationError()],
                            ),
            MinSaver('validation_error'), #save model with min val-error, must be after inference
            #ScheduledHyperParamSetter('learning_rate',
            #                          [(args.drop_0, args.scale_lr*args.lr_0),
            #                           (args.drop_1,  args.scale_lr*args.lr_1)]),
            #HyperParamSetterWithFunc('learning_rate',
            #                         lambda e, x: x * float(0.1) if e % 15 == 0 and e > args.drop_2 else x),# (1+e)/(2*20) #ScheduledHyperParamSetter('learning_rate',[(args.drop_0, args.scale_lr*args.lr_0), (args.drop_1,  args.scale_lr*args.lr_1), (args.drop_2,  args.scale_lr*args.lr_2), (args.drop_3,  args.scale_lr*args.lr_3)]), # denote current hyperparameter)
            StatMonitorParamSetter('learning_rate', 'validation_error',
                                   lambda x: x * 0.1, threshold=1e-15, last_k=20),
            MergeAllSummaries()
         ],
         model = denseModel,
         session_creator = None,
         session_config = train_config,
         steps_per_epoch=steps_per_epoch,
         max_epoch=args.max_epoch,
      )
   else:
      """
      Predictive model configuration for testing 
      and classifying.
      """
      class TestParamSetter(Callback):
         #def _before_run(self, _):
         #   return tf.train.SessionRunArgs(fetches=[],feed_dict={'PlaceholderWithDefault_1:0':1.0, 'PlaceholderWithDefault_2:0':False})#'drop_rate:0':1, 'train_or_test:0':False
         def _setup_graph(self):
            self._drop_rate = [k for k in tf.global_variables() if k.name == 'PlaceholderWithDefault_1:0'][0]
            self._train_or_test = [k for k in tf.global_variables() if k.name == 'PlaceholderWithDefault_2:0'][0]
         def _trigger_step(self):
            self._drop_rate.load(1.0)
            self._train_or_test.load(False)
      print(">>>>>> Constructing prediction variables.")
      return PredictConfig(
         model = denseModel,
         input_names= ['input', 'label'],#denseModel._get_inputs(),
         output_names=['output', 'train_error', 'cross_entropy_loss', 'input'],
      )

class predictModel:
   def __init__(self, config, data = None,  num_gpu = 0, image_size = None):
      self.test_data = data
      self.predictor = SimpleDatasetPredictor(config, self.test_data)
      #MultiProcessDatasetPredictor(config, self.test_data, nr_proc = num_gpu, use_gpu = True, ordered = True)
      
   def _get_inputs(self):
      return [InputDesc(tf.float32, [None, args.image_size, args.image_size, 3], 'input'),
              InputDesc(tf.int32, [None], 'label')
      ]
   
   def get_results(self):
      res = []
      all_res = []
      # Output for each datapoint in dataflow
      for prob, training_error, cross_entropy, img in self.predictor.get_result():
         res.append( ["element", prob, training_error, cross_entropy, img] )# tf.argmax(prob,1)
      for prob, training_error, cross_entropy, img in self.predictor.get_all_result():
         all_res.append(  ["element", prob , training_error, cross_entropy, img] )
      return res, all_res


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--model_name', type=str, default='MODEL', help="Name to prepend on model during training")
   parser.add_argument('--gpu', default=None, help='comma separated list of GPU(s) to use.')
   parser.add_argument('--load', help='load model')
   parser.add_argument('--kernels',type=int, default=24, help='initial kernel channel depth. Number kernels.')
   parser.add_argument('--kernel_size',type=int, default=3, help='initial kernel window size')
   parser.add_argument('--expansion', type=int, default=12,help='expansion growth rate of kernels between layers per convolution')
   parser.add_argument('--drop_0',type=int, default=1, help='Epoch to drop learning rate to lr_0.')
   parser.add_argument('--drop_1',type=int, default=80, help='Epoch to drop learning rate to lr_1.')
   parser.add_argument('--drop_2',type=int, default=150,help='Epoch to drop learning rate to lr_2.')
   parser.add_argument('--drop_3',type=int, default=150,help='Epoch to drop learning rate to lr_3.')
   parser.add_argument('--lr_init',type=float, default=0.001,help='starting learning rate')
   parser.add_argument('--lr_0',type=float, default=0.0001,help='second learning rate')
   parser.add_argument('--lr_1',type=float, default=0.00001, help='first learning rate')             
   parser.add_argument('--lr_2',type=float, default=0.000001,help='second learning rate')
   parser.add_argument('--lr_3',type=float, default=0.0000001,help='third learning rate')
   parser.add_argument('--class_weights',type=str,help='comma seperated class weights e.g. class 0, class 1')
   parser.add_argument('--batch_size',type=int, default=4,help='batch size')
   parser.add_argument('--depth',type=int, default=13, help='The depth of densenet')
   parser.add_argument('--drop_out',type=float, help='drop out rate')
   parser.add_argument('--drop_pattern',type=int, default=0, help='drop layer every drop_pattern layer')
   parser.add_argument('--skip_norm',type=int, default=None, help='Do not use batchnorm every skip_norm layer e.g. (layer%skip_norm).')
   parser.add_argument('--bn_momentum',type=float, default=None, help='momentum for batch norm during inference.')
   parser.add_argument('--max_epoch',type= int, default=256,help='max epoch')
   parser.add_argument('--tot',type= str, default='train',help=" 'train' or 'test'")
   parser.add_argument('--out_dir',type= str, default='../data/Unknowns/predictions/',help="img out dir")
   parser.add_argument('--unknown_dir',type= str, default='../data/Unknowns/predictions/',help="unknown samples to classify")
   parser.add_argument('--save_sample',type= bool, default=False,help="boolean save originals")
   parser.add_argument('--original_dir',default=False,help="directory to save originals")
   parser.add_argument('--data_dir', default=os.getcwd()+'/data', help="directory to read data from")
   parser.add_argument('--num_gpu',type= int,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--scale_lr',type= float,default=1.0,help="Scale learning rate factor (for distributed training)")
   parser.add_argument('--gpu_frac',type= float,default=0.96,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--mp',type=int, default=2,help="Whether or not to use parallel multiprocessing over or on GPU. 0 no, 1 yes. Default yes.")
   parser.add_argument('--nccl',type=int,default=1,help="Whether NCCL available for DGX like machines.")
   parser.add_argument('--class_0',type=int, default=1,help="number samples in class 0")
   parser.add_argument('--class_1',type=int, default=1,help="number samples in class 1")
   parser.add_argument('--multi_crop',type=int, default=4,help="number, if any, of crops to take from crop_per_case images. Average of crop classifications used in application")
   parser.add_argument('--image_size',type=int, default=224,help="original image size to scale to scale_size ")
   parser.add_argument('--aug_norm',type=int, default=0,help="pre-process image augmentation by normalizing H&E staining. 1 yes, 0 no.")
   parser.add_argument('--aug_randnorm',type=float, default=0.0,help="Normalization of H&E staining image augmentation performed with probability of value passed range [0,1.0)")
   parser.add_argument('--aug_randhe',type=float, default=0.0,help="perform random augmentation of H&E staining color space image augmentation with probability passed [0,1.0)")
   parser.add_argument('--aug_he',type=int, default=0,help="image augmentation by randomly permuting H&E staining. 1 yes, 0 no.")
   parser.add_argument('--scale_size',type=int, default=224,help="image size after scaling")
   parser.add_argument('--scale',type=int, default=2,help="factor of 224 to crop and scale down scale*(224,224)->(224,224)")
   parser.add_argument('--crop_per_case',type=int, default=40,help="number, more than 1, of images from each case to multi_crop. Average of crop classifications used in application")
   parser.add_argument('--unique_samples',type=int, help="Number of unique samples (excluding crops) used for averaging testing results")
   parser.add_argument('--frac_res', type=float, default=1.0, help="fraction to reduce resolution of data used for training")

   args = parser.parse_args()
   args.scale_size = args.image_size
   
   if args.class_weights:
      args.class_weights = [float(weight) for weight in args.class_weights.split(',')]
   
   if args.gpu or (not args.num_gpu):
      #from tensorpack.utils.gpu import get_num_gpu
      args.num_gpu = len(args.gpu.split(','))#max(get_num_gpu(),1)
      note = "Slurm assigns on DGX"
      print("...")
      print(">>>> NOTE!")
      print(" >>>> Hard assigning to all available gpu")
      os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
      os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu#','.join(map(str,range(args.num_gpu)))#args.gpu
   
   if not args.tot:
      args.tot == 'train'
      
   session_config = None
   if True:#args.tot =='train':
      print("Using personal session config")
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_frac, allow_growth=True)
      session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement = True) 
      
   nr_tower = 1
   if args.gpu or args.num_gpu:
      if args.num_gpu and (not args.gpu):
         print(">>>> Using "+str(args.num_gpu)+ " available GPU.")
         nr_tower = args.num_gpu
      else:
         nr_tower = len(args.gpu.split(','))
   
   config = get_config(args.tot, train_config=session_config, load_model = args.load)
   #print(tf.test.is_gpu_available())                                          
   #print(get_available_gpus())
   
   print("Net configured")
   
   if args.load:
      print(">>>> Loading stored model parameters.")
      # example args.load '/path/to/model/folder/model-xxxx'
      config.session_init = SaverRestore(args.load)
   
   if args.tot == 'train':
      if args.mp==0:
         print("using simple trainer")
         launch_train_with_config(config, SimpleTrainer())
      else:
         print("can use simple (mp=0) trainer multi gpu parameter server or replicated")
         print("for nccl as well as multiprocess distributed (mp=2) or multithreaded distributed (mp=else)")
         if args.nccl == 0:
            print(">>>> Using "+str(args.num_gpu)+" available GPU parameter server.")
            launch_train_with_config(config, SyncMultiGPUTrainer(args.num_gpu))
         elif args.num_gpu and args.nccl != 0:
            print(">>>> Using "+str(args.num_gpu)+" available GPU for replicated training (nccl).")
            launch_train_with_config(config, SyncMultiGPUTrainerReplicated(args.num_gpu))
   else:
      # retrieve unknown from /data/Unknown/unknown_dir   optional save data to original_dir (creates if d.n.e.)
      # writes predictions (image and summary file) to path = '../data/Unknown/'+args.unknown_dir +'/predictions'
      
      data = get_data('test',image_size=args.image_size, scale_size = args.scale_size, scale = args.scale, multi_crop = args.multi_crop, crop_per_case = args.crop_per_case, normalize = args.aug_norm, shuffle = False,unknown_dir = args.unknown_dir, original_dir=args.original_dir)
      
      def classify_unknown(model_config, dp, args):
         if args.gpu:
            args.num_gpu = len(args.gpu.split(',')) if args.num_gpu is None else args.num_gpu
         
         predictor = predictModel(config, data = dp, num_gpu = args.num_gpu, image_size = args.image_size)
         res, all_res = predictor.get_results()
         img_list = []
         prediction = []
         BL_count = 0
         DLBCL_count = 0
         tie = 0
         path = ''
         multi_crop_count = 0
         crop_per_case=1
         max_crop_met = 0
         
         crop_preds_bl = 0.0
         crop_preds_dl = 0.0
         if bool(args.multi_crop) != False:
            multi_crop_count = 0
            crop_per_case = min(args.unique_samples, args.crop_per_case)
            
         for i, op in enumerate(res[0][4]):
            prediction.append(res[0][1][i])
            img_list.append(op)
            
            #crop_preds_bl = 0.0
            #crop_preds_dl = 0.0
            
            if args.multi_crop==0 or args.crop_per_case == 0:
               max_crop_met += 1
            elif (i+1)/args.multi_crop >= args.crop_per_case:
               max_crop_met += 1
            
            path = os.path.join(args.data_dir,'Unknown',args.unknown_dir)#predictions/'+args.out_dir+'_Predictions/'
            if not os.path.exists( os.path.join(path, 'predictions')):
               os.makedirs( os.path.join(path, 'predictions') )
            path =  os.path.join(path, 'predictions')
            
            im = Image.fromarray(np.asarray(op).astype('uint8'),'RGB')#.astype('uint8'))                                    
            im.save(path + '/' +str(i)+"_"+str(res[0][1][i])+'.jpeg')
            im.close()
            
            multi_crop_count += 1
            crop_preds_bl+=float(res[0][1][i][0])
            crop_preds_dl+=float(res[0][1][i][1])
            
            if multi_crop_count==args.multi_crop or max_crop_met > 0:
               if max_crop_met > 1:
                  multi_crop = 1.0
               else:
                  multi_crop = args.multi_crop if args.multi_crop != 0 else 1.0
               
               pred_bl = crop_preds_bl/float(multi_crop)
               pred_dl = crop_preds_dl/float(multi_crop)
               crop_preds_bl = 0.0
               crop_preds_dl = 0.0
               
               if pred_bl > 0.5:
                  BL_count += 1
               if pred_dl > 0.5:
                  DLBCL_count += 1
               if pred_bl == pred_dl:
                  tie += 1
               
               multi_crop_count = 0 if max_crop_met == 0 else -1
               crop_per_case -= 1
               
               path = os.path.join(args.data_dir,'Predictions',args.unknown_dir)
               if not os.path.exists( path ):
                  os.makedirs( path )
               
               path = path+'/res80'+args.unknown_dir+'.txt'
               if os.path.exists(path):
                  append_write = 'a'
                  #write results to file                   
               else:
                  append_write = 'w'
               prediction_file = open(path, append_write)
               prediction_file.write(str(pred_bl)+','+str(pred_dl)+"\n")
               prediction_file.close()
               
         prediction_summary = open( path, 'a')
         #'../data/Unknown/'+args.unknown_dir+'/predictions/predictions.txt', 'a')
         prediction_summary.write("\n")
         if tie:
            prediction_summary.write(str(BL_count)+','+str(DLBCL_count)+"\n")
            prediction_summary.write(str(tie)+"\n")
         else:
            prediction_summary.write(str(BL_count)+','+str(DLBCL_count)+"\n")
         prediction_summary.close()
         
      if args.batch_size == 1:
         classify_unknown(config, data, args)
      else:
         classify_unknown(config, data, args)
