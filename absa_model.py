import os
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange

class MemN2N_ABSA(object):
    def __init__(self, config, sess, embed_nparray, labels_dict):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        # mem_size is also the size of the mask
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        # sentiment analysis parameters
        self.nlabels = config.nlabels
        self.embed_nparray = embed_nparray
        self.labels_dict = labels_dict

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        # input is the aspect here
        #self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.input = tf.placeholder(tf.int32, [None], name="input")
        self.input = tf.Print(self.input, [tf.shape(self.input)], "shape self.input: ", summarize=5, first_n=2)

        #self.mem_embd = tf.Variable(
        self.mem_embd = tf.get_variable(
	    name='embedding_weights',
            shape=(self.nwords, self.edim),
            initializer=tf.constant_initializer(self.embed_nparray),
            trainable=True
        )

        self.embed_input = tf.nn.embedding_lookup(self.mem_embd, self.input)

        # target is sentiment class in the case below
        self.target = tf.placeholder(tf.float32, [None, self.nlabels], name="target")
        self.target = tf.Print(self.target, [tf.shape(self.target)], "shape self.target: ", summarize=5, first_n=2)

        # mem_size is different form edim in that mem_size is the max sent
        # len; edim is the embedding dimension (say glove)
        self.context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")
        self.context = tf.Print(self.context, [tf.shape(self.context)], "shape self.context: ", summarize=5, first_n=2)

        self.mask = tf.placeholder(tf.float32, [None, self.mem_size], name="context")
        self.mask = tf.Print(self.mask, [tf.shape(self.mask)], "shape self.mask: ", summarize=5, first_n=2)

        self.hid = []
        #self.hid.append(self.input)
        self.hid.append(self.embed_input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        print "self.nwords: ", self.nwords

        # memory = mem_dim(d) x context_len
        # stack each memory with aspect to get 2d x context_len
        # multiple it with W_att (1x2d) to get 1 x context_len
        # pass it through nn.tanh

        # cntxt_size is mem_size
        #self.W_att = tf.Variable(tf.random_uniform(shape=[self.mem_size, \
        # TODO: research very surprising that same W_att is used for all memory
        # words
        self.W_att = tf.Variable(tf.random_uniform(shape=[2*self.edim, 1], \
            minval=-0.01, maxval=0.01))
        self.b_att = tf.Variable(tf.random_uniform(shape=[], minval=-0.01, maxval=0.01))
        self.W_lin = tf.Variable(tf.random_uniform(shape=[1, self.edim, self.edim], \
                minval=-0.01, maxval=0.01)) # ignorign multi-word aspect

        self.b_sftmx = tf.Variable(tf.random_uniform(shape=[1, self.nlabels], minval=-0.01, maxval=0.01))
        self.W_sftmx = tf.Variable(tf.random_uniform(shape=[self.edim, self.nlabels], \
                minval=-0.01, maxval=0.01)) # ignorign multi-word aspect

        cntxt_raw = tf.nn.embedding_lookup(self.mem_embd, self.context)
        # the above dimension batch_sz x num_cntxt_wrds (mem_size) x edim
        # TODO apply mask above -- done-- check
        cntxt_out = tf.multiply(cntxt_raw, tf.expand_dims(self.mask, 2))
        cntxt_out = tf.Print(cntxt_out, [tf.shape(cntxt_out)], "shape cntxt_out: ", summarize=5, first_n=2)

        for h in xrange(self.nhop):
            # TODO: put the mask in -- ni parts this assumes that puting mask in
            # stops the gradient flow. -- verify this.
            # linear layer weight has to be tiled in order to accomodate batch
            W_lin_tile = tf.tile(self.W_lin, [tf.shape(self.hid[-1])[0], 1, 1])
            # the above is batch_size x edim x edim
            lyr_in = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            lin_out = tf.matmul(lyr_in, W_lin_tile) # shape = batch_size x 1 x edim
            lyr_in_linr_part = tf.reshape(lin_out, [-1, self.edim])
            lyr_in_linr_part = tf.Print(lyr_in_linr_part, [tf.shape(lyr_in_linr_part)], \
                "shape lyr_in_linr_part: ", summarize=5, first_n=2)
            # shape above batch_size x edim
            # below concatenate cntxt_mem and aspect
            # We will have to first tile aspect
            # Then concatenate with context; this is all assuming input is
            # batch_size x edim and context is batch_size x mem_size  x edim
            inpt_shp = tf.shape(self.hid[-1])
            inpt_3d = tf.reshape(self.hid[-1], [inpt_shp[0], 1, inpt_shp[1]])
            inpt_tld = tf.tile(inpt_3d, [1, self.mem_size, 1])
            cnctnted_raw = tf.concat([cntxt_out, inpt_tld], axis=2)
            cnctnted = tf.multiply(cnctnted_raw, tf.expand_dims(self.mask, 2))
            cnctnted = tf.Print(cnctnted, [tf.shape(cnctnted)], "shape cnctnted: ", summarize=5, first_n=2)
            # the above is batch_size x mem_size x 2*edim
            # TODO: apply mask here above. done
            w_att_tld = tf.tile(tf.reshape(self.W_att, [1, 2*self.edim, 1]), \
                    [inpt_shp[0], 1, 1])
            w_att_tld = tf.Print(w_att_tld, [tf.shape(w_att_tld)], "shape w_att_tld: ", summarize=5, first_n=2)
            g_in_mulprt = tf.matmul(cnctnted, w_att_tld)
            g_in_mulprt = tf.Print(g_in_mulprt, [tf.shape(g_in_mulprt)], "shape g_in_mulprt: ", summarize=5, first_n=2)
            g_in = tf.add(g_in_mulprt, self.b_att)
            g_in = tf.Print(g_in, [tf.shape(g_in)], "shape g_in: ", summarize=5, first_n=2)
            g_out = tf.nn.tanh(g_in) # shape = batch_size x mem_size x 1
            alpha_in_raw = tf.reshape(g_out, [-1, 1, self.mem_size])
            alpha_in_raw = tf.Print(alpha_in_raw, [tf.shape(alpha_in_raw)], "shape alpha_in_raw: ", summarize=5, first_n=2)
            #NOTE: we dont need reshape above: https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
            #alpha_in = tf.multiply(alpha_in_raw, tf.expand_dims(self.mask, 2))
            alpha_in = tf.multiply(alpha_in_raw, tf.reshape(self.mask, [-1, 1, self.mem_size]))
            alpha_in = tf.Print(alpha_in, [tf.shape(alpha_in)], "shape alpha_in: ", summarize=5, first_n=2)
            alpha_out = tf.nn.softmax(alpha_in) # shape = batch_size x 1 x mem_size
            #TODO: apply mask above - done. we went aggressive with mask applying
            alphaXmem = tf.matmul(alpha_out, cntxt_out)
            alphaXmem = tf.Print(alphaXmem, [tf.shape(alphaXmem)], "shape \
                    alphaXmem: ", summarize=5, first_n=2)
            # above shape = batch_size x 1 x edim
            lyr_in_mempart = tf.reshape(alphaXmem, [-1, self.edim])
            lyr_in_mempart = tf.Print(lyr_in_mempart, [tf.shape(lyr_in_mempart)], \
                "shape lyr_in_mempart: ", summarize=5, first_n=2)
            lyr_out = tf.add(lyr_in_linr_part, lyr_in_mempart)
            self.hid.append(lyr_out)


    def build_model(self):
        self.build_memory()

        #self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        #z = tf.matmul(self.hid[-1], self.W)

        z = self.hid[-1]
        z_logit = tf.add(tf.matmul(z, self.W_sftmx), self.b_sftmx)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z_logit, labels=self.target)
        self.num_accurate = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(z_logit),\
                tf.argmax(self.target)), tf.float32))

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        #params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        params = [self.W_att, self.W_lin, self.W_sftmx, self.b_att, self.b_sftmx]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, cnsldted_data):
        (data, trgt_aspect, trgt_Y, cntxt_mask) = cnsldted_data
        # the data above is context not input; trgt_aspect is input (x below)
        # the datastructs above assume consistent sequential storage
        N = int(math.ceil(len(data) / self.batch_size))
        # many sentences are repeated due to multiple aspects in a single
        # sentence.
        cost = 0
        accurate = 0

        #x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        x = np.ndarray([self.batch_size], dtype=np.int32)
        # the above is input array
        target = np.zeros([self.batch_size, self.nlabels]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])
        mask = np.ndarray([self.batch_size, self.mem_size])

        #x.fill(self.init_hid)
        #for t in xrange(self.mem_size):
        #    time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N):
            # wE arte doing sequenctial batching here.
            if self.show: bar.next()
            target.fill(0)
            curr_indx = idx*self.batch_size
            for b in xrange(self.batch_size):
                senti_labl = trgt_Y[curr_indx+b]
                target[b][self.labels_dict[senti_labl]] = 1
                context[b] = data[curr_indx+b]
                mask[b] = cntxt_mask[curr_indx+b]
                x[b] = trgt_aspect[curr_indx+b]
                #TODO: fill the x -- DONE
                # above assuming that each element of the list data is a list of
                # words in original sentence

            _, loss, self.step, num_accurate = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step,
                                                self.num_accurate],
                                                feed_dict={
                                                    self.input: x,
                                                    #self.time: time,
                                                    self.target: target,
                                                    self.context: context,
                                                    self.mask: mask})
            cost += np.sum(loss)
            accurate += num_accurate

        if self.show: bar.finish()
        return cost/N/self.batch_size, accurate*1./N/self.batch_size

    def test(self, cnsldted_data, label='Test'):
        (data, trgt_aspect, trgt_Y, cntxt_mask) = cnsldted_data
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0
        accurate = 0

        x = np.ndarray([self.batch_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nlabels]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])
        mask = np.ndarray([self.batch_size, self.mem_size])

        #x.fill(self.init_hid)
        #for t in xrange(self.mem_size):
        #    time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            curr_indx = idx*self.batch_size
            for b in xrange(self.batch_size):
                senti_labl = trgt_Y[curr_indx+b]
                target[b][self.labels_dict[senti_labl]] = 1
                context[b] = data[curr_indx+b]
                mask[b] = cntxt_mask[curr_indx+b]
                x[b] = trgt_aspect[curr_indx+b]
                # above assuming that each element of the list data is a list of
                # words in original sentence

            loss, num_accurate = self.sess.run([self.loss, self.num_accurate], feed_dict={self.input: x,
                                                         #self.time: time,
                                                         self.target: target,
                                                         self.context: context,
                                                         self.mask: mask})
            cost += np.sum(loss)
            accurate += num_accurate

        if self.show: bar.finish()
        return cost/N/self.batch_size, accurate*1./N/self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                #train_loss, train_accuracy = np.sum(self.train(train_data))
                train_loss_list, train_accuracy = self.train(train_data)
                train_loss = np.sum(train_loss_list)
                test_loss_list, test_accuracy = self.test(test_data, label='Validation')
                test_loss = np.sum(test_loss_list)

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'accuracy': train_accuracy,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss),
                    'test_accuracy': test_accuracy
                }
                print(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()

            #valid_loss, valid_accuracy = np.sum(self.test(train_data, label='Validation'))
            valid_loss_list, valid_accuracy = self.test(train_data, label='Validation')
            valid_loss = np.sum(valid_loss_list)
            test_loss_list, test_accuracy = self.test(test_data, label='Test')
            test_loss = np.sum(test_loss_list)

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'valid_accuracy': valid_accuracy,
                'test_perplexity': math.exp(test_loss),
                'test_accuracy': test_accuracy
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
