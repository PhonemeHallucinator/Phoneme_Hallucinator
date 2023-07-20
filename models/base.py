import logging
from pprint import pformat
import numpy as np
import tensorflow as tf
class BaseModel(object):
    def __init__(self, hps):
        super(BaseModel, self).__init__()

        self.hps = hps
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build model
            self.build_net()
            self.build_ops()
            # initialize
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.hps.exp_dir + '/summary')
            # logging
            total_params = 0
            trainable_variables = tf.trainable_variables()
            logging.info('=' * 20)
            logging.info("Variables:")
            logging.info(pformat(trainable_variables))
            for v in trainable_variables:
                num_params = np.prod(v.get_shape().as_list())
                total_params += num_params

            logging.info("TOTAL TENSORS: %d TOTAL PARAMS: %f[M]" % (
                len(trainable_variables), total_params / 1e6))
            logging.info('=' * 20)

    def save(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.save(self.sess, fname)

    def load(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.restore(self.sess, fname)

    def build_net(self):
        raise NotImplementedError()

    def build_ops(self):
        # optimizer
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.inverse_time_decay(
            self.hps.lr, self.global_step,
            self.hps.decay_steps, self.hps.decay_rate,
            staircase=True)
        warmup_lr = tf.train.inverse_time_decay(
            0.001 * self.hps.lr, self.global_step,
            self.hps.decay_steps, self.hps.decay_rate,
            staircase=True)
        learning_rate = tf.cond(tf.less(self.global_step, 1000), lambda: warmup_lr, lambda: learning_rate)
        tf.summary.scalar('lr', learning_rate)
        if self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name="Adam")
        elif self.hps.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        # regularization
        l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
        reg_loss = 0.00005 * l2_reg

        # train
        grads_and_vars = optimizer.compute_gradients(
            self.loss+reg_loss, tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        if self.hps.clip_gradient > 0:
            grads, gradient_norm = tf.clip_by_global_norm(
                grads, clip_norm=self.hps.clip_gradient)
            gradient_norm = tf.check_numerics(
                gradient_norm, "Gradient norm is NaN or Inf.")
            tf.summary.scalar('gradient_norm', gradient_norm)
        capped_grads_and_vars = zip(grads, vars_)
        self.train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=self.global_step)
        
        # summary
        self.summ_op = tf.summary.merge_all()

    def execute(self, cmd, batch):
        return self.sess.run(cmd, {self.x:batch['x'], self.b:batch['b'], self.m:batch['m']})
