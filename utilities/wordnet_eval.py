# Copyright 2016 Mandiant, A FireEye Company
# Authors: Brian Jones
# License: Apache 2.0

''' Example run script for "Relational Learning with TensorFlow" tutorial '''

import os
from pprint import pprint
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

import utilities.models as models
import utilities.util as util


#######################################
# Data preprocessing

def read_wordnet(fpath, def_df):
    df = pd.read_table(fpath, names=['head', 'rel', 'tail'])
    df['head'] = def_df.loc[df['head']]['word'].values
    df['tail'] = def_df.loc[df['tail']]['word'].values
    return df


def wordnet_preprocess(train, val, test):
    mask = np.zeros(len(train)).astype(bool)
    lookup = defaultdict(list)
    for idx,h,_,t in train.itertuples():
        lookup[(h,t)].append(idx) 
    for h,_,t in pd.concat((val,test)).itertuples(index=False):
        mask[lookup[(h,t)]] = True
        mask[lookup[(t,h)]] = True
    train = train.loc[~mask]
    heads, tails = set(train['head']), set(train['tail'])
    val = val.loc[val['head'].isin(heads) & val['tail'].isin(tails)]
    test = test.loc[test['head'].isin(heads) & test['tail'].isin(tails)]
    return train, val, test


#######################################
# Models used in tutorial

def cp():
    opt = tf.train.AdagradOptimizer(1.0)
    return models.Contrastive_CP(embedding_size=20, 
                                 maxnorm=1.5,
                                 batch_pos_cnt=100, 
                                 max_iter=30000, 
                                 model_type='least_squares',
                                 add_bias=False,
                                 opt=opt)


def bilinear():
    opt = tf.train.AdagradOptimizer(1.0)
    return models.Bilinear(embedding_size=20, 
                           maxnorm=1.0,
                           rel_maxnorm_mult=6.0,
                           batch_pos_cnt=100, 
                           max_iter=30000, 
                           model_type='logistic',
                           add_bias=True,
                           opt=opt)


def transe():
    opt = tf.train.AdagradOptimizer(1.0)
    return models.TransE(embedding_size=20,
                         batch_pos_cnt=100,
                         max_iter=30000,
                         dist='euclidean',
                         margin=1.0,
                         opt=opt)
    

if __name__ == '__main__':
    
    ###################################
    # MODEL
    
    rng = np.random.RandomState(123)
    model = transe() #transe() # bilinear() # cp()

    print(model.__class__)
    pprint(model.__dict__)
    
    ###################################
    # DATA
    
    data_dir = '../data/wordnet-mlj12'
    definitions = pd.read_table(os.path.join(data_dir, 'wordnet-mlj12-definitions.txt'), 
                                index_col=0, names=['word', 'definition'])
    train = read_wordnet(os.path.join(data_dir, 'wordnet-mlj12-train.txt'), definitions)
    val = read_wordnet(os.path.join(data_dir, 'wordnet-mlj12-valid.txt'), definitions)
    test = read_wordnet(os.path.join(data_dir, 'wordnet-mlj12-test.txt'), definitions)
    combined_df = pd.concat((train, val, test))
    all_train_entities = set(train['head']).union(train['tail'])
    all_train_relationships = set(train['rel'])
    
    print()
    print('Train shape:', train.shape)
    print('Validation shape:', val.shape)
    print('Test shape:', test.shape)
    print('Training entity count: {}'.format(len(all_train_entities)))
    print('Training relationship type count: {}'.format(len(all_train_relationships)))
    
    print()
    print('Preprocessing to remove instances from train that have a similar counterpart in val/test...')
    train,val,test = wordnet_preprocess(train, val, test)
    all_train_entities = set(train['head']).union(train['tail'])
    all_train_relationships = set(train['rel'])
    
    print('Adding negative examples to val and test...')
    combined_df = pd.concat((train, val, test))
    val = util.create_tf_pairs(val, combined_df, rng)
    test = util.create_tf_pairs(test, combined_df, rng)
    print('Train shape:', train.shape)
    print('Validation shape:', val.shape)
    print('Test shape:', test.shape)
    print()

    if isinstance(model, models.Contrastive_CP):
        print('Using separate encoding for head and tail entities')
        field_categories = (set(train['head']), 
                            all_train_relationships,
                            set(train['tail']))
    else:
        print('Using the same encoding for head and tail entities')
        field_categories = (all_train_entities, 
                            all_train_relationships,
                            all_train_entities)

    train, train_idx_array = util.make_categorical(train, field_categories)
    val, val_idx_array = util.make_categorical(val, field_categories)
    test, test_idx_array = util.make_categorical(test, field_categories)
    print('Train check:', train.shape, not train.isnull().values.any())
    print('Val check:', val.shape, not val.isnull().values.any())
    print('Test check:', test.shape, not test.isnull().values.any())

    ###################################
    # TRAIN
    
    # Monitor progress on current training batch and validation set
    start = time.time()
    val_labels = np.array(val['truth_flag'], dtype=np.float)
    val_feed_dict = model.create_feed_dict(val_idx_array, val_labels)
    
    def train_step_callback(itr, batch_feed_dict):
        if (itr % 2000) == 0 or (itr == (model.max_iter-1)):
            elapsed = int(time.time() - start)
            avg_batch_loss = model.sess.run(model.loss, batch_feed_dict) / len(batch_feed_dict[model.target])
            avg_val_loss = model.sess.run(model.loss, val_feed_dict) / len(val_labels)
            val_acc = util.model_pair_ranking_accuracy(model, val_idx_array)
            msg = 'Itr {}, train loss: {:.3}, val loss: {:.3}, val rank_acc: {:.2}, elapsed: {}'
            print(msg.format(itr, avg_batch_loss, avg_val_loss, val_acc, elapsed))
            # Check embedding norms
            names,model_vars = zip(*model.embeddings())
            var_vals = model.sess.run(model_vars)
            for name,var in zip(names, var_vals):
                norms = np.linalg.norm(var, axis=1)
                print('{} min/max norm: {:.2} {:.2}'.format(name, np.min(norms), np.max(norms)))
        return True

    print('Training...')
    model.fit(train_idx_array, train_step_callback)

    ###################################
    # TEST
    
    print()
    print('Done training, evaluating on test set.')
    test_labels = np.array(test['truth_flag'], dtype=np.float)
    test_feed_dict = model.create_feed_dict(test_idx_array, test_labels, training=False)
    acc, pred, scores, thresh_map = util.model_threshold_and_eval(model, test, val)
    
    print('Test set accuracy: {:.2}'.format(acc))
    print()
    print('Relationship breakdown:')
    results_df = test.copy()
    results_df['score'] = scores
    results_df['prediction'] = pred
    results_df['is_correct'] = pred == test['truth_flag']
    for rel in set(results_df['rel']):
        rows = results_df[results_df['rel'] == rel]
        n = len(rows)
        correct = rows['is_correct'].sum()
        wrong = n - correct
        print('acc:{:.2} rel:{}, {} / {}'.format(float(correct)/n, rel, correct, n))
