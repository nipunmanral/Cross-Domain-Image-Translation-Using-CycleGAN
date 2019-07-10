import numpy as np
import glob
from scipy.misc import imread, imresize
import tensorflow as tf
import matplotlib.pyplot as plt

def load_train_images(data_dir):
    images_type_A = glob.glob(data_dir + '/trainA/*.jpg')
    images_type_B = glob.glob(data_dir + '/trainB/*.jpg')

    processed_imagesA = []
    processed_imagesB = []

    for i, filename in enumerate(images_type_A):
        imA = imread(filename, mode='RGB')
        imB = imread(images_type_B[i], mode='RGB')

        imA = imresize(imA, (128, 128))
        imB = imresize(imA, (128, 128))

        #Randomly flip some images
        if np.random.random() > 0.5:
            imA = np.fliplr(imA)
            imB = np.fliplr(imB)
        
        processed_imagesA.append(imA)
        processed_imagesB.append(imB)

    #Normalise image values between -1 and 1
    processed_imagesA = np.array(processed_imagesA)/127.5 - 1.0
    processed_imagesB = np.array(processed_imagesB)/127.5 - 1.0

    return processed_imagesA, processed_imagesB

def load_test_images(data_dir, num_images):
    images_type_A = glob.glob(data_dir + '/testA/*.jpg')
    images_type_B = glob.glob(data_dir + '/testB/*.jpg')

    images_type_A = np.random.choice(images_type_A, num_images)
    images_type_B = np.random.choice(images_type_B, num_images)

    processed_imagesA = []
    processed_imagesB = []

    for i in range(len(images_type_A)):
        imA = imresize(imread(images_type_A[i], mode='RGB').astype(np.float32), (128, 128))
        imB = imresize(imread(images_type_B[i], mode='RGB').astype(np.float32), (128, 128))

        processed_imagesA.append(imA)
        processed_imagesB.append(imB)

    #Normalise image values between -1 and 1
    processed_imagesA = np.array(processed_imagesA)/127.5 - 1.0
    processed_imagesB = np.array(processed_imagesB)/127.5 - 1.0

    return processed_imagesA, processed_imagesB

#Save the training losses to the tensorboard logs that can be used for visualization
def save_losses_tensorboard(callback, name, loss, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

def save_test_results(realA, realB, fakeA, fakeB, reconsA, reconsB, identityA, identityB):
    for i in range(len(realA)):
        fig = plt.figure()            
        plt.imshow(realA[i])
        plt.axis('off')
        plt.savefig("results/m2f/real_{}".format(i), bbox_inches='tight')
        fig2 = plt.figure()
        plt.imshow(fakeB[i])
        plt.axis('off')
        plt.savefig("results/m2f/fake_{}".format(i), bbox_inches='tight')
        fig3 = plt.figure()
        plt.imshow(reconsA[i])
        plt.axis('off')
        plt.savefig("results/m2f/recons_{}".format(i, bbox_inches='tight'))
        fig4 = plt.figure()
        plt.imshow(identityA[i])
        plt.axis('off')
        plt.savefig("results/m2f/identity_{}".format(i), bbox_inches='tight')
        fig = plt.figure()            
        plt.imshow(realB[i])
        plt.axis('off')
        plt.savefig("results/f2m/real_{}".format(i), bbox_inches='tight')
        fig2 = plt.figure()
        plt.imshow(fakeA[i])
        plt.axis('off')
        plt.savefig("results/f2m/fake_{}".format(i), bbox_inches='tight')
        fig3 = plt.figure()
        plt.imshow(reconsB[i])
        plt.axis('off')
        plt.savefig("results/f2m/recons_{}".format(i), bbox_inches='tight')
        fig4 = plt.figure()
        plt.imshow(identityB[i])
        plt.axis('off')
        plt.savefig("results/f2m/identity_{}".format(i), bbox_inches='tight')
