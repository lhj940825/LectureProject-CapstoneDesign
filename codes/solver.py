import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from random import random
from PIL import Image
import cv2 as cv

#
# K = 5749
# N = 13233

i = 0


face_count = 0
# # 100 사용시
file_n = 56
classes = 100

xs = np.load('face_data/matrix.npy')    # 이미지 matrix (152, 152, 3)
#ys = np.load('face_data/one_hot.npy')   # one hot 라벨 (5749, 1)
zs = np.load('face_data/index.npy')   # one hot 라벨 (5749, 1)


class Solver(object):
        #############################################################################################################################
   # def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100,
             #    svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample',
              #   model_save_path='model', pretrained_model='model/svhn_model-20000', test_model='model/dtn-1800'):
    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100,
                svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample',
                model_save_path='model_resize', pretrained_model='model_resize/svhn_model-5000', test_model='model_resize/dtn-2000'):
        ###################################################################################################################################
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

    def load_svhn(self, image_dir, split='train'):
        print ('loading svhn image dataset..')
        
        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1 #정규화
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        print ('finished loading svhn image dataset..!')
        return images, labels

    def load_svhn_testing(self, image_dir, size=[152, 152, 3], split='train'):
        print('loading svhn image dataset..')

        image_file = 'extra_32x32.mat'
        print(image_dir)
        image_dir = os.path.join(image_dir, image_file)
        print(image_dir)
        svhn = scipy.io.loadmat(image_dir)

        svhn['X'] = np.transpose(svhn['X'], [3, 0, 1, 2])
        svhn['X'] = svhn['X'][:1000, :, :, :]  # (1000,152,152,3)

        resized_image_arrays = np.zeros([svhn['X'].shape[0]] + size)

        for i, image_array in enumerate(svhn['X']):
            image = Image.fromarray(image_array)
            resized_image = image.resize(size=[152, 152], resample=Image.ANTIALIAS)
            resized_image_arrays[i] = np.asarray(resized_image)

        images = resized_image_arrays / 127.5 - 1  # 정규화

        print(np.shape(resized_image_arrays[0:9, :, :, :]))
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        print('finished loading svhn image dataset..!')

        return images, labels
        #return xs[:100], zs[:100]

    def load_face(self):
        # rs = []
        #
        # for _ in range(100):
        #     rs.append(int(random() * N)) # K => 13233
        #
        # def load_mat(idx):
        #     return xs[idx]
        #
        # def load_idx(idx):
        #     return zs[idx]
        #
        # xss = list(map(load_mat, rs))
        # zss = list(map(load_idx, rs))
        # # print(xss[0], zss[0], rs)
        # #Input normalization
        # return np.array(xss)/127.5 -1, np.array(zss)

        global face_count
        file_path = os.path.join('face_loader', 'mats', str(classes))
        cycle = str(face_count % file_n).zfill(4)
        x = 'x' + cycle + '.npy'
        y = 'y' + cycle + '.npy'
        x_path = os.path.join(file_path, x)
        y_path = os.path.join(file_path, y)
        face_count += 1
        return np.load(x_path)/127.5 -1, np.load(y_path)


    def load_caricature(self):
        global i
        idx = i % 1000
        filename = 'cmats/' + str(idx).zfill(4) + '.npy'
        i += 1
        # (100, 152, 152, 3)
        xss = np.load(filename)/127.5 -1
        return xss

    def load_mnist(self, image_dir, split='train'):
        print ('loading mnist image dataset..')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print ('finished loading mnist image dataset..!')
        return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        # cv.imshow("",merged)
        # cv.waitKey(0)
        return merged

    def pretrain(self):
        # load svhn dataset
        ##train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        ##test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        ####################################################################################
        #train_images, train_labels = self.load_svhn_testing(self.svhn_dir, split='train')
        #test_images, test_labels = self.load_svhn_testing(self.svhn_dir, split='test')

        # 얼굴 데이터로 입력 수정

        #train_images, train_labels = self.load_face()
        #test_images, test_labels = self.load_face()
        ####################################################################################

        # build a graph
        model = self.model
        model.build_model(self.batch_size)

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                #i = step % int(train_images.shape[0] / self.batch_size)
                #batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                #batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size]

                train_images, train_labels = self.load_face()



                feed_dict = {model.images: train_images, model.labels: train_labels}


                sess.run(model.train_op, feed_dict)
                #tmp = model.labels.eval(feed_dict=feed_dict)
                #print('label'+str(tmp))
                #print("label="+ batch_labels)
                if (step+1) % 100 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    # rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    # test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                    #                        feed_dict={model.images: test_images[rand_idxs],
                    #                                   model.labels: test_labels[rand_idxs]})
                    # summary_writer.add_summary(summary, step)
                    # print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                    #            %(step+1, self.pretrain_iter, l, acc, test_acc))
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [0]' \
                                %(step+1, self.pretrain_iter, l, acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1) 
                    print ('svhn_model-%d saved..!' %(step+1))

    def train(self):
        print("load_dataset")

        # build a graph
        model = self.model
        model.build_model(self.batch_size)

        print("build_dataset_complete")

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F

            print ('loading pretrained model F..')
            # 부분적으로 restore!
            # variables_to_restore = slim.get_model_variables(scope='content_extractor')
            # restorer = tf.train.Saver(variables_to_restore)
            # restorer.restore(sess, self.pretrained_model)
            # summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            # saver = tf.train.Saver()

            #
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)



            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):

                # load svhn dataset
                ##################################################################################
                # svhn_images, _ = self.load_svhn(self.svhn_dir, split='train')

                ##################################################################################
                face_image, _ = self.load_face()




                # train the model for source domain S


                
                #sess.run(model.d_train_op_src, feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                # sess.run([model.g_train_op_trg], feed_dict)
                
                #if step > 1600:
                #    f_interval = 30
                
                # if i % f_interval == 0:
                #     sess.run(model.f_train_op_src, feed_dict)
                

                
                # train the model for target domain T

                src_images = face_image
                trg_images = self.load_caricature()
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}

                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)


                # if (step+1) % 10 == 0:
                #     summary, dl, gl = sess.run([model.summary_op_trg, \
                #         model.d_loss_trg, model.g_loss_trg], feed_dict)
                #     summary_writer.add_summary(summary, step)
                #     print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                #                %(step+1, self.train_iter, dl, gl))

                if (step+1) % 100 == 0:
                        dl, gl= sess.run([\
                         model.D_loss, model.G_loss], feed_dict)
                     #summary_writer.add_summary(summary, step)
                        print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                                %(step+1, self.train_iter, dl, gl))

                if (step + 1) % 100 == 0:
                    feed_dict4Sample = {model.src_images: src_images}

                    sample_images = sess.run(model.src_images_FG, feed_dict4Sample)
                    sample_images = (sample_images + 1) * 127.5
                    src_images = (src_images + 1) * 127.5
                    merged = self.merge_images(src_images, sample_images)
                    path = os.path.join(self.sample_save_path,
                                        'sample-step-%d.jpeg' % (step + 1))
                    cv.imwrite(path, merged)
                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print ('model/dtn-%d saved' %(step+1))

                
    def eval(self):

        # build model
        model = self.model
        model.build_model(self.batch_size)

        # load svhn dataset


        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):















                # train model for source domain S
                batch_images, _  = self.load_face()
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)
                print("image")


                sampled_batch_images=(sampled_batch_images+1)*127.5
                batch_images = (batch_images+1)*127.5
                merged = self.merge_images(batch_images, sampled_batch_images)

                #print(str((sampled_batch_images)))
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.jpeg' %(i*self.batch_size, (i+1)*self.batch_size))
                # cv.imshow(str(merged.shape), merged)
                # cv.waitKey(0)


                cv.imwrite(path, merged)
                # scipy.misc.imsave(path, merged)
                print ('saved %s' %path)



# eval 함수 내용
#
# # train model for source domain S
# batch_images, _ = self.load_face()
# feed_dict = {model.images: batch_images}
# sampled_batch_images = sess.run(model.sampled_images, feed_dict)
# print("image")
#
# sampled_batch_images = (sampled_batch_images + 1) * 127.5
# batch_images = (batch_images + 1) * 127.5
# merged = self.merge_images(batch_images, sampled_batch_images)
#
# # print(str((sampled_batch_images)))
# path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.jpeg' % (i * self.batch_size, (i + 1) * self.batch_size))
# # cv.imshow(str(merged.shape), merged)
# # cv.waitKey(0)
#
#
# cv.imwrite(path, merged)
# # scipy.misc.imsave(path, merged)
# print('saved %s' % path)