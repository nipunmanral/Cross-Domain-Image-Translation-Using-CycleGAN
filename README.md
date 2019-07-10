# Gender Change of People's Face using CycleGAN
## Summary
We implement the CycleGAN architecture in Keras and train the model with CelebA faces dataset to perform gender change on people's faces. There are two main scripts in the code - `predict.py` and `train.py`

## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.6 is installed in the current environment. Then execute 

    pip install -r requirements.txt

This should install all the necessary packages for the code to run.

The data used in this project is obtained from the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and it has to be saved in the folder structure described below. For training, we have to pre-process the images by resizing and removing bad quality photos. For testing, the code automatically handles the conversion of the image files into appropriate size. The generated images will be 128x128 RGB images.

## Training 

By default, you can put the data in **Code/data/male_female/** directory. The training and test data should be provided in 4 seperate directories as: 
* trainA - images of male faces for training
* trainB - images of female faces for training
* testA - images of male faces for testing
* testB - images of female faces for testing

Run the code as 

    python train.py

If the data is in a different directory, you can specify the path at runtime by using the *--data_dir* flag:

    python train.py --data_dir <path/to/data>

To see the parameters that can be changed, run

    python train.py -h

The training loss will be updated in the logs folder which can be run with tensorboard to visualise the generator and discriminator loss on the browser. Run this command:

    tensorboard --logdir=logs

At the end of training, the output of this code is the model weights of the two generators and the two discriminators. These will be saved as:
* generatorAToB.h5 - 43.5MB
* generatorBToA.h5 - 43.5MB
* discriminatorA.h5 - 10.5MB
* discriminatorB.h5 - 10.5MB

## Testing
Once the model is trained, the model file should be in the same folder as the test script with the name: *generatorAToB.h5* and *generatorBToA.h5*. The test data should be in *--data_dir* parameter in the folder structure as described above. Using the *--batchSize* flag, you can specify home many test images should get modified.

You can test the model by running:

    python predict.py

The results will be generated in *Code/results* folder. Create two folders *m2f* and *f2m* inside the results folder. The corresponding transformations will appear in these two folders. The results will contain - fake, real, reconstructed and identity images.

## Video
Project Video: https://www.youtube.com/watch?v=PgPfN3v4lG4&t=404s
