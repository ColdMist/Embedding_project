#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:55:46 2018

@author: turzo
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display

from utilities import util

def read_and_replace(fpath, def_df):
    df = pd.read_table(fpath, names=['head', 'rel', 'tail'])
    df['head'] = def_df.loc[df['head']]['word'].values
    df['tail'] = def_df.loc[df['tail']]['word'].values
    return df

#Load the data and create a pandas dataframe object
data_dir = 'data/wordnet-mlj12' # change to where you extracted the data
definitions = pd.read_table(os.path.join(data_dir, 'wordnet-mlj12-definitions.txt'), 
                            index_col=0, names=['word', 'definition'])
train = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-train.txt'), definitions)
val = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-valid.txt'), definitions)
test = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-test.txt'), definitions)

#Look at the data to get an overview
print('Train shape:', train.shape)
print('Validation shape:', val.shape)
print('Test shape:', test.shape)
all_train_entities = set(train['head']).union(train['tail'])
print('Training entity count: {}'.format(len(all_train_entities)))
print('Training relationship type count: {}'.format(len(set(train['rel']))))
print('Example training triples:')
display(train.sample(5))

'''
Since most of the relation in the triples of the validation set and test sets are mirror images of the training set, that is
why we are going to remove that too.
'''
#we will find the statndat nonlinearity based on the function
from collections import defaultdict

#Now create a mask varialbe for removing purpose
mask = np.zeros(len(train)).astype(bool)
lookup = defaultdict(list)

#iterate through the training set and append the head and tail 
for idx,h,r,t in train.itertuples():
    lookup[(h,t)].append(idx) # try to use lookup default dictionary to append the indexes to the head, tail as index
#Combine the training validation and test set using concat fucntion of dataframe
#train_val_test_combined = pd.concat((train,val,test))
#Build the mask with the indicies of the validation and test set so that we can remove them from training set
for h,r,t in pd.concat((val,test)).itertuples(index=False):
    mask[lookup[(h,t)]] = True
    mask[lookup[(t,h)]] = True
#Remove the masked rows from the training set
train = train.loc[~mask]
heads,tails = set(train['head']), set(train['tail'])
val = val.loc[val['head'].isin(heads) & val['tail'].isin(tails)]
test = test.loc[test['head'].isin(heads) & test['tail'].isin(tails)]
'''
We will now creates some false statemens to make it as a classification problem [Socher13]. For each true statement corrupt it by either 
either replacing the head or tail with a random checking.
'''
#Here we will use our own create corrupt sample creation

rng = np.random.RandomState(42)
combined_df = pd.concat((train, val, test))
val = util.create_tf_pairs(val, combined_df, rng)
test = util.create_tf_pairs(test, combined_df, rng)
print('Validation shape:', val.shape)
print('Test shape:', test.shape)
#For testing purpose
#rng = np.random.RandomState(42)
#combined_df = pd.concat((train, val, test))

#val_res = util.create_tf_pairs(val, combined_df, rng)
#test_res = util.create_tf_pairs(test, combined_df, rng)

#Lets check what kind of prediction task we are up against, lets examine the training and test data for involving the entity 'brain_cell'
example_entity = '__brain_cell_NN_1'
example_train_rows = (train['head'] == example_entity) | (train['tail'] == example_entity)
print('Train: ')
display(train.loc[example_train_rows])
example_test_rows = (test['head'] == example_entity) | (test['tail'] == example_entity)
print('Test: ')
display(test.loc[example_test_rows])


has_part_triples = val.loc[val['rel'] == '_has_part']
query_entities = ['__noaa_NN_1', '__vascular_plant_NN_1', '__retina_NN_1']
has_part_example = has_part_triples.loc[has_part_triples['head'].isin(query_entities)]
matrix_view = pd.pivot_table(has_part_example, 'truth_flag', 'head', 'tail',
                             fill_value=False).astype(int)
display(matrix_view)

#Right now we will create the input and the place holders
graph = tf.Graph()
with graph.as_default():
    head_input = tf.placeholder(tf.int32, shape=[None])
    rel_input = tf.placeholder(tf.int32, shape = [None])
    tail_input = tf.placeholder(tf.int32, shape = [None])
    target = tf.placeholder(tf.float32, shape = [None])

embedding_size = 20
head_cnt = len(set(train['head'])) #Count the number of heads in the training set
rel_cnt = len(set(train['rel'])) #Count the number of relations in the training set
tail_cnt = len(set(train['tail'])) #Count the number of tail in the training set

#One hot encode each of the three inputs for head, rel, tail into long vectors so that only the corresponding item is set to one.
#Three input vectors should be connected to the embedding layer in fully connected faction as typical neural networks
#Weight matrix would contain the embedding for each item.


embedding_size = 20
head_cnt = len(set(train['head']))
rel_cnt = len(set(train['rel']))
tail_cnt = len(set(train['tail']))

with graph.as_default():
    # embedding variables
    init_sd = 1.0 / np.sqrt(embedding_size)
    head_embedding_vars = tf.Variable(tf.truncated_normal([head_cnt, embedding_size],
                                                          stddev=init_sd))
    rel_embedding_vars = tf.Variable(tf.truncated_normal([rel_cnt, embedding_size],
                                                         stddev=init_sd))
    tail_embedding_vars = tf.Variable(tf.truncated_normal([tail_cnt, embedding_size],
                                                          stddev=init_sd))
    # embedding layer for the (h, r, t) triple being fed in as input
    head_embed = tf.nn.embedding_lookup(head_embedding_vars, head_input)
    rel_embed = tf.nn.embedding_lookup(rel_embedding_vars, rel_input)
    tail_embed = tf.nn.embedding_lookup(tail_embedding_vars, tail_input)
    # CP model output
    output = tf.reduce_sum(tf.multiply(tf.multiply(head_embed, rel_embed), tail_embed), 1)

# TensorFlow requires integer indices
field_categories = (set(train['head']), set(train['rel']), set(train['tail']))
train, train_idx_array = util.make_categorical(train, field_categories)
val, val_idx_array = util.make_categorical(val, field_categories)
test, test_idx_array = util.make_categorical(test, field_categories)

###########################################################################################################################################
from utilities.util import ContrastiveTrainingProvider

batch_provider = ContrastiveTrainingProvider(train_idx_array, batch_pos_cnt=3,
                                             separate_head_tail=True)
batch_triples, batch_labels = batch_provider.next_batch()
batch_df = pd.DataFrame()
batch_df['head'] = pd.Categorical.from_codes(batch_triples[:,0], train['head'].cat.categories)
batch_df['rel'] = pd.Categorical.from_codes(batch_triples[:,1], train['rel'].cat.categories)
batch_df['tail'] = pd.Categorical.from_codes(batch_triples[:,2], train['tail'].cat.categories)
batch_df['label'] = batch_labels
display(batch_triples)
print('which encodes:')
display(batch_df)

max_iter = 30000

batch_provider = ContrastiveTrainingProvider(train_idx_array, batch_pos_cnt=100,
                                             separate_head_tail=True)
opt = tf.train.AdagradOptimizer(1.0)

sess = tf.Session(graph=graph)
with graph.as_default():
    loss = tf.reduce_sum(tf.square(output - target))
    train_step = opt.minimize(loss)
    sess.run(tf.initialize_all_variables())

# feed dict for monitoring progress on validation set
val_labels = np.array(val['truth_flag'], dtype=np.float)
val_feed_dict = {head_input: val_idx_array[:,0],
                 rel_input: val_idx_array[:,1],
                 tail_input: val_idx_array[:,2],
                 target: val_labels}

for i in range(max_iter):
    batch_triples, batch_labels = batch_provider.next_batch()
    feed_dict = {head_input: batch_triples[:,0],
                 rel_input: batch_triples[:,1],
                 tail_input: batch_triples[:,2],
                 target: batch_labels}
    if i % 2000 == 0 or i == (max_iter-1):
        batch_avg_loss = sess.run(loss, feed_dict) / len(batch_labels)
        val_output, val_loss = sess.run((output,loss), val_feed_dict)
        val_avg_loss = val_loss / len(val_labels)
        val_pair_ranking_acc = util.pair_ranking_accuracy(val_output)
        msg = 'iter {}, batch loss: {:.2}, val loss: {:.2}, val true false ranking acc: {:.2}'
        print(msg.format(i, batch_avg_loss, val_avg_loss, val_pair_ranking_acc))
    sess.run(train_step, feed_dict)
# the session will be closed
sess.close()



