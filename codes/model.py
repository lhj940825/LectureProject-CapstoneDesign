import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import get_variables
from utils import *



class DTN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.01):
        self.mode = mode
        self.learning_rate = learning_rate

    # TODO CNN 구조 변경(convolution) 및 shape 복사 삭제
    def content_extractor(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)


        with tf.variable_scope('content_extractor', reuse=reuse):
            # slim.conv2d에 대한 arg들을 설정!
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=1,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train' or self.mode=='pretrain')):

                    print(str(np.shape(images)))
                    net = slim.conv2d(images, 16, [11, 11], scope='conv1', padding='VALID', activation_fn=tf.nn.relu)   # (batch_size, 142, 142, 32)
                    print("conv1" + str(np.shape(net)))
                    #net = slim.batch_norm(net, scope='bn1')
                    net = slim.max_pool2d(net, [3,3], stride= 2, padding='SAME') # (batch_size, 71, 71, 32)
                    print("max_pool1" + str(np.shape(net)))
                    net = slim.conv2d(net, 32, [9, 9], scope='conv2', padding='VALID', activation_fn=tf.nn.relu)     # (batch_size, 63, 63, 16)
                    print("conv2" + str(np.shape(net)))
                    net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME')  # (batch_size, 32, 32, 16)
                    print("max_pool2" + str(np.shape(net)))
                    #net = slim.batch_norm(net, scope='bn2')
                    #print("batch2" + str(np.shape(net)))
                    net = slim.conv2d(net, 64, [9, 9], scope='conv3', padding='VALID', activation_fn=tf.nn.relu)     # (batch_size, 24, 24, 16)
                    print("conv3" + str(np.shape(net)))
                    #net = slim.batch_norm(net, scope='bn3')
                    #print("batch3" + str(np.shape(net)))
                    net = slim.conv2d(net, 100, [7, 7], scope='conv4', padding='VALID', activation_fn=tf.nn.relu)   # (batch_size, 18, 18, 16)
                    print("conv4" + str(np.shape(net)))
                    net = slim.conv2d(net, 128, [5, 5], scope='conv5', padding='VALID',
                                      activation_fn=tf.nn.relu)  # (batch_size, 14, 14, 16)
                    print("conv5" + str(np.shape(net)))
                    net = slim.conv2d(net, 256, [3, 3], scope='conv6', padding='VALID', stride=2,
                                      activation_fn=tf.nn.relu)  # (batch_size, 6, 6, 16)
                    print("conv6" + str(np.shape(net)))
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')
                    print("batch4" + str(np.shape(net))) # (batch_size, 6, 6, 256)
                    if self.mode == 'pretrain':
                        net = tf.reshape(net, [-1, 9216]) # (batch_size, 2304)
                        net = slim.dropout(net, 0.5, scope='dropout' )
                        net = slim.fully_connected(net, 500) # (batch_size, 800)
                        print('fc1'+str(np.shape(net)))
                        net = slim.fully_connected(net, 100) # (batch_size, 100)
                        print('fc2'+str(np.shape(net)))

                    return net

                # def content_extractor(self, images, reuse=False):
                #     # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
                #
                #     if images.get_shape()[3] == 1:
                #         # For mnist dataset, replicate the gray scale image 3 times.
                #         images = tf.image.grayscale_to_rgb(images)
                #
                #     with tf.variable_scope('content_extractor', reuse=reuse):
                #         # slim.conv2d에 대한 arg들을 설정!
                #         with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                #                             stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                #             with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                #                                 activation_fn=tf.nn.relu,
                #                                 is_training=(self.mode == 'train' or self.mode == 'pretrain')):
                #                 print(str(np.shape(images)))
                #                 net = slim.conv2d(images, 64, [2, 2], scope='conv1')  # (batch_size, 16, 16, 64)
                #                 print("conv1" + str(np.shape(net)))
                #                 net = slim.batch_norm(net, scope='bn1')
                #                 print("batch1" + str(np.shape(net)))
                #                 net = slim.conv2d(net, 128, [2, 2], scope='conv2')  # (batch_size, 8, 8, 128)
                #                 print("conv2" + str(np.shape(net)))
                #                 net = slim.batch_norm(net, scope='bn2')
                #                 print("batch2" + str(np.shape(net)))
                #                 net = slim.conv2d(net, 256, [2, 2], scope='conv3')  # (batch_size, 4, 4, 256)
                #                 print("conv3" + str(np.shape(net)))
                #                 net = slim.batch_norm(net, scope='bn3')
                #                 print("batch3" + str(np.shape(net)))
                #                 net = slim.conv2d(net, 128, [2, 2], padding='VALID',
                #                                   scope='conv4')  # (batch_size, 1, 1, 128)
                #                 print("conv4" + str(np.shape(net)))
                #                 net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')
                #                 print("batch4" + str(np.shape(net)))
                #                 if self.mode == 'pretrain':
                #                     net = slim.conv2d(net, 5749, [1, 1], padding='VALID', scope='out')
                #                     net = slim.flatten(net)
                #                 return net

    # TODO CNN 구조 변경(Deconvolution)
    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128), Consist of 9 layers
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    print("Ginput" + str(np.shape(inputs)))

                    # Block 1
                    net = slim.conv2d_transpose(inputs, 512, [5, 5],  activation_fn=tf.nn.relu, padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    print("Gconv1" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [1, 1], stride = 1, padding='SAME',
                                                scope='1x1conv_transpose1')
                    # Block 2
                    net = slim.conv2d_transpose(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='VALID', scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    print("Gconv2" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 96, [1, 1], stride=1, padding='SAME',
                                      scope='1x1conv_transpose2')

                    # Block 3
                    net = slim.conv2d_transpose(net, 64, [2, 2], activation_fn=tf.nn.relu, padding='VALID', scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    print("Gconv3" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 32, [1, 1], stride=1, padding='SAME',
                                      scope='1x1conv_transpose3')

                    # Block 4
                    net = slim.conv2d_transpose(net, 16, [2, 2],activation_fn=tf.nn.relu, padding='VALID', stride=1, scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    print("Gconv4" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn4')
                    net = slim.conv2d(net, 8, [1, 1], stride=1, padding='SAME',
                                      scope='1x1conv_transpose4')
                    # Block 5

    #                    net = slim.batch_norm(net, scope='bn5')
                    net = slim.conv2d_transpose(net, 3, [2, 2], padding='VALID', stride=1, activation_fn=tf.nn.tanh,
                                                scope='conv_transpose5')  # (batch_size, 32, 32, 3)
                    net = slim.dropout(net, 0.5, scope='G_dropout')
                    print("Gconv5" + str(np.shape(net)))
                    return net

    #TODO images.get_shape() 부분 삭제
    def discriminator(self, images, reuse=False):

        # if images.get_shape()[3] == 1:
        #     # For mnist dataset, replicate the gray scale image 3 times.
        #     images = tf.image.grayscale_to_rgb(images)
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    # images (batch_size, 152, 152, 3)

                    # Block 1
                    net = slim.conv2d(images, 3, [3, 3], scope='conv1')   # (batch_size, 76, 76, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    print("Disconv1" + str(np.shape(net)))


                    # Block 2
                    net = slim.conv2d(net, 16, [3, 3], scope='conv2')   # (batch_size, 38, 38, 256)
                    print("Disconv2" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn2')

                    # Block 3
                    net = slim.conv2d(net, 32, [3, 3], scope='conv3')   # (batch_size, 19, 19, 512)
                    print("Disconv3" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn3')

                    # Block 4
                    net = slim.conv2d(net, 56, [2, 2], padding='VALID', scope='conv4')   # (batch_size, 9, 9, 256)
                    print("Disconv4" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn4')
                    net = slim.conv2d(net, 76, [1, 1], stride= 1, scope="1x1conv4")

                    # Block 5
                    net = slim.conv2d(net, 64, [2, 2], padding='VALID', scope='conv5')   # (batch_size, 4, 4, 64)
                    print("Disconv5" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn5')
                    net = slim.conv2d(net, 128, [1, 1], stride= 1, scope="1x1conv5")

                    # Block 6
                    net = slim.conv2d(net, 156, [2, 2], padding='VALID', scope='conv6')   # (batch_size, 2, 2, 16)
                    print("Disconv6" + str(np.shape(net)))
                    net = slim.batch_norm(net, scope='bn6')



                    net = tf.reshape(net, [-1, 624])  # (batch_size, 576)
                    net = slim.fully_connected(net, 56)  # (batch_size, 5749)
                    net = slim.dropout(net, 0.4, scope='D_dropout')
                    net = slim.fully_connected(net, 3)  # (batch_size, 5749)
                    #net = slim.flatten(net) # (batch_size, 3)
                    print("dis output" + str(np.shape(net)))

                    return net
                
    def build_model(self, batch_size):
        self.batch_size = batch_size
        if self.mode == 'pretrain':
            #########################################################################################
            self.images = tf.placeholder(tf.float32, [None, 152, 152, 3], 'svhn_images')
            #########################################################################################
            #self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')


            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)

            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op

            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'eval':
            #########################################################################################
            self.images = tf.placeholder(tf.float32, [None, 152, 152, 3], 'svhn_images')
            #########################################################################################

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)
            self.sampled_images = tf.image.resize_images(self.sampled_images, [152,152])

        elif self.mode == 'train':
            self.CONST_weight = 15
            self.TID_weight = 15
            self.TV_weight = 1

            print("in_model")
            #########################################################################################
            self.src_images = tf.placeholder(tf.float32, [None, 152, 152, 3], 'face_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 152, 152, 3], 'caricature_images')
            #########################################################################################


            self.src_images_F = self.content_extractor(self.src_images)

            self.src_images_FG = self.generator(self.src_images_F)
            self.src_images_FG = tf.image.resize_images(self.src_images_FG, [152,152])
            self.src_images_FGF = self.content_extractor(self.src_images_FG, reuse=True)


            self.trg_images_F = self.content_extractor(self.trg_images, reuse=True)
            self.trg_images_FG = self.generator(self.trg_images_F, reuse=True)
            self.trg_images_FG = tf.image.resize_images(self.trg_images_FG, [152,152])

            self.D_logits_1 = self.discriminator(self.src_images_FG)
            self.D_logits_2 = self.discriminator(self.trg_images_FG, reuse=True)
            self.D_logits_3 = self.discriminator(self.trg_images, reuse = True)

            self.D_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits_1, labels=np.array([np.array([1.0, 0.0, 0.0]) for i in range(batch_size)])))
            self.D_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits_2, labels=
                                                                                   np.array(
                                                                                       [np.array([0.0, 1.0, 0.0]) for i
                                                                                        in range(self.batch_size)])))
            self.D_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits_3, labels=
                                                                                   np.array(
                                                                                       [np.array([0.0, 0.0, 1.0]) for i
                                                                                        in range(self.batch_size)])))

            self.D_loss = self.D_loss_1 + self.D_loss_2 + self.D_loss_3

            self.GANG_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits_1, labels=
                                                                                    np.array(
                                                                                        [np.array([0.0, 0.0, 1.0]) for i
                                                                                         in range(self.batch_size)]))) \
                             + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits_2,labels=
                                                                                      np.array(
                                                                                          [np.array([0.0, 0.0, 1.0]) for
                                                                                           i in
                                                                                           range(self.batch_size)])))

            self.CONST_loss = tf.reduce_mean(tf.squared_difference(self.src_images_F, self.src_images_FGF))
            self.TID_loss = tf.reduce_mean(tf.squared_difference(self.trg_images, self.trg_images_FG))
            self.G_loss = self.GANG_loss + self.CONST_weight * self.CONST_loss + self.TID_weight * self.TID_loss

            t_vars = tf.trainable_variables()

            self.D_vars = get_variables(scope="discriminator")
            self.G_vars = get_variables(scope="generator")

            self.saver = tf.train.Saver(var_list=self.D_vars + self.G_vars)

            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            #
            # # train op
            # # with tf.name_scope('target_train_op'):
            with tf.variable_scope('target_train_op', reuse=False):
                self.d_train_op_trg = self.d_optimizer_trg.minimize(self.D_loss, var_list= self.D_vars)
                self.g_train_op_trg = self.d_optimizer_trg.minimize(self.G_loss, var_list= self.G_vars)

            print((self.D_logits_1))

            #
            # # source domain (svhn to mnist)
            # self.fx = self.content_extractor(self.src_images)
            # self.fake_images = self.generator(self.fx)
            # self.logits = self.discriminator(self.fake_images)
            # self.fgfx = self.content_extractor(self.fake_images, reuse=True) #f(g(f(x)))를 의미
            #
            # # loss
            # self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            # self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            # self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0
            #
            #
            #
            # # optimizer
            # self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            #
            #
            #
            # t_vars = tf.trainable_variables()
            # d_vars = [var for var in t_vars if 'discriminator' in var.name]
            # g_vars = [var for var in t_vars if 'generator' in var.name]
            # f_vars = [var for var in t_vars if 'content_extractor' in var.name]
            #
            #
            # # tf.reset_default_graph()
            #
            # # train op
            # # with tf.name_scope('source_train_op'):
            # print("in_model2")
            # with tf.variable_scope('source_train_op',reuse=False):
            #     self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
            #     self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
            #     self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=f_vars)
            #
            # # summary op
            # d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            # g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            # f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            # origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            # sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            # self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary,
            #                                         f_loss_src_summary, origin_images_summary,
            #                                         sampled_images_summary])
            #
            # # target domain (mnist)
            # self.fx = self.content_extractor(self.trg_images, reuse=True)
            # self.reconst_images = self.generator(self.fx, reuse=True)
            # self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            # self.logits_real = self.discriminator(self.trg_images, reuse=True)
            #
            # # loss
            # self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            # self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            # self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            # self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            # self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * 15.0
            # self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg
            #

            #
            # # summary op
            # d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            # d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            # d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            # g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            # g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            # g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            # origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            # sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            # self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary,
            #                                         d_loss_fake_trg_summary, d_loss_real_trg_summary,
            #                                         g_loss_fake_trg_summary, g_loss_const_trg_summary,
            #                                         origin_images_summary, sampled_images_summary])
            # print("in_model3")
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)
            # print("in_model4")
