import numpy as np
import pickle
import os
import scipy.io
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data



def resize_images(image_arrays, size=[152, 152]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)
        
        resized_image_arrays[i] = np.asarray(resized_image)
    print(np.shape(resized_image_arrays))
    print(np.shape(np.expand_dims(resized_image_arrays, 3)))
    return np.expand_dims(resized_image_arrays, 3)  


def load_svhn_testing(image_dir, size = [152,152,3], split='train'):
    print('loading svhn image dataset..')


    image_file = 'extra_32x32.mat'


    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)

    svhn['X'] = np.transpose(svhn['X'], [3, 0, 1, 2])
    svhn['X'] = svhn['X'][:1000,:,:,:] # (1000,152,152,3)

    resized_image_arrays = np.zeros([svhn['X'].shape[0]] + size)


    for i , image_array in enumerate(svhn['X']):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=[152,152], resample=Image.ANTIALIAS)
        resized_image_arrays[i] = np.asarray(resized_image)


    images = resized_image_arrays/ 127.5 - 1  # 정규화

    print(np.shape(resized_image_arrays[0:9,:,:,:]))
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    print('finished loading svhn image dataset..!')
    return images, labels

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    mnist = input_data.read_data_sets(train_dir='mnist2')

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}
    
    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}
        
    save_pickle(train, 'mnist2/train.pkl')
    save_pickle(test, 'mnist2/test.pkl')
    
    
if __name__ == "__main__":
    main()
    