import numpy as np
import tensorflow as tf

def layer_norm(x):
    mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    std = tf.math.reduce_std(x, axis=[1,2], keepdims=True)
    x = (x - mean) / std
    return x

def set_attention(Q, K, dim, num_heads, name='set_attention'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        q = tf.layers.dense(Q, dim, name='query')
        k = tf.layers.dense(K, dim, name='key')
        v = tf.layers.dense(K, dim, name='value')

        q_ = tf.concat(tf.split(q, num_heads, axis=-1), axis=0)
        k_ = tf.concat(tf.split(k, num_heads, axis=-1), axis=0)
        v_ = tf.concat(tf.split(v, num_heads, axis=-1), axis=0)

        logits = tf.matmul(q_, k_, transpose_b=True)/np.sqrt(dim) # [B*Nh,Nq,Nk]
        A = tf.nn.softmax(logits, axis=-1)
        o = q_ + tf.matmul(A, v_)
        
        o = tf.concat(tf.split(o, num_heads, axis=0), axis=-1)
        # o = tf.contrib.layers.layer_norm(o)
        o = o + tf.layers.dense(o, dim, activation=tf.nn.relu, name='output')
        # o = tf.contrib.layers.layer_norm(o)

    return o

def set_transformer(inputs, layer_sizes, name, num_heads=4, num_inds=16):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            inds = tf.get_variable(f'inds_{i}', shape=[1,num_inds,size], dtype=tf.float32, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
            inds = tf.tile(inds, [tf.shape(out)[0],1,1])
            tmp = set_attention(inds, out, size, num_heads, name=f'self_attn_{i}_pre')
            out = set_attention(out, tmp, size, num_heads, name=f'self_attn_{i}_post')

    return out

def set_pooling(inputs, name, num_heads=4):
    B = tf.shape(inputs)[0]
    N = tf.shape(inputs)[1]
    d = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        seed = tf.get_variable('pool_seed', shape=[1,1,d], dtype=tf.float32, trainable=True, 
                                initializer=tf.contrib.layers.xavier_initializer())
        seed = tf.tile(seed, [B,1,1])
        out = set_attention(seed, inputs, d, num_heads, name='pool_attn')
        out = tf.squeeze(out, axis=1)

    return out
