''' CMoDE:  Adaptive  Semantic  Segmentation
              in  Adverse  Environmental  Conditions
 Copyright (C) 2018  Abhinav Valada, Johan Vertens , Ankit Dhall and Wolfram Burgard
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import numpy as np
import tensorflow as tf
import network_base
from AdapNet_pp import AdapNet_pp
from AdapNet import AdapNet

class CMoDE(network_base.Network):
    def __init__(self, num_classes=12, learning_rate=0.001, float_type=tf.float32, weight_decay=0.0005,
                 decay_steps=30000, power=0.9, training=True, ignore_label=True, global_step=0,
                 has_aux_loss=False, expert_not_fixed=False, mode='AdapNet_pp'):
        super(CMoDE, self).__init__()
        if mode == 'AdapNet_pp':
            self.model1 = AdapNet_pp(training=expert_not_fixed, num_classes=num_classes) 
            self.model2 = AdapNet_pp(training=expert_not_fixed, num_classes=num_classes) 
        else:
            self.model1 = AdapNet(training=expert_not_fixed, num_classes=num_classes) 
            self.model2 = AdapNet(training=expert_not_fixed, num_classes=num_classes) 
        self.mode = mode
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initializer = 'he'
        self.has_aux_loss = has_aux_loss
        self.float_type = float_type
        self.power = power
        self.decay_steps = decay_steps
        self.training = training
        self.bn_decay_ = 0.99
        self.eAspp_rate = [3, 6, 12]
        self.residual_units = [3, 4, 6, 3]
        self.filters = [256, 512, 1024, 2048]
        self.strides = [1, 2, 2, 1]
        self.global_step = global_step
        if self.training:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0

        if ignore_label:
            self.weights = tf.ones(self.num_classes-1)
            self.weights = tf.concat((tf.zeros(1), self.weights), 0)
        else:
            self.weights = tf.ones(self.num_classes)

    def _setup(self):   
        if self.mode == 'AdapNet_pp':
            self.in1 = tf.concat([self.model1.eAspp_out, self.model2.eAspp_out], 3)
        else:
            self.in1 = tf.concat([self.model1.m_b4_out, self.model2.m_b4_out], 3)
        count = 511
        out = []
        out1 = []
        for i in xrange(self.num_classes):
            temp = self.spatial_dropout(tf.nn.relu(self.conv_bias(self.in1, 3, 1, 20, 'conv'+str(count))), self.keep_prob)
            count = count+1
            temp1 = tf.nn.relu(self.fc(tf.reshape(temp, [tf.shape(temp)[0], 24*48*20]), 128, name='fc'+str(count)))
            count = count+1
            temp2 = tf.nn.softmax(self.fc(temp1, 2, name='fc'+str(count)))
            count = count+1
            o, p = tf.split(temp2, 2, 1)
            out.append(o)
            out1.append(p)

        prob = tf.expand_dims(tf.expand_dims(tf.squeeze(tf.stack(out, axis=1), axis=2), axis=1), axis=2)
        prob1 = tf.expand_dims(tf.expand_dims(tf.squeeze(tf.stack(out1, axis=1), axis=2), axis=1), axis=2)
        self.in2 = tf.add(tf.multiply(self.model1.deconv_up3, prob1), tf.multiply(self.model2.deconv_up3, prob))
        self.in5 = self.conv_bias(self.in2, 3, 1, self.num_classes, name=('conv%d'%count))
        self.softmax = tf.nn.softmax(self.in5)

    def _create_loss(self, label):
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.softmax+1e-10), self.weights), axis=[3]))

    def create_optimizer(self):
        self.lr = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                            self.decay_steps, power=self.power)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self, data, data1, label=None):
        with tf.variable_scope('rgb/resnet_v1_50'):
            self.model1.build_graph(data)
        with tf.variable_scope('depth/resnet_v1_50'):
            self.model2.build_graph(data1)
        self._setup()
        if self.training:
            self._create_loss(label)
            
    def spatial_dropout(self,x, keep_prob):
        num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(num_feature_maps,
                                       dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        binary_tensor = tf.reshape(binary_tensor, 
                               [-1, 1, 1, tf.shape(x)[3]])
        ret = tf.div(x, keep_prob) * binary_tensor
        return ret

     
def main():
    print 'Do Nothing'

if __name__ == '__main__':
    main()
