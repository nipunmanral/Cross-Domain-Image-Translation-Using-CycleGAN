from utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchSize', type=int, default=1, help='Batch Size to be used for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs that training should run')
    parser.add_argument('--lambda_cyc', type=int, default=10, help='lambda value for cycle consistency loss')
    parser.add_argument('--lambda_idt', type=int, default=5, help='lambda value for identity loss')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='The frequency at which model should be saved and evaluated')
    parser.add_argument('--num_resnet_blocks', type=int, default=9, help='Number of ResNet blocks for transformation in generator')
    parser.add_argument('--data_dir', type=str, default='data/male_female/', help='Directory where train and test images are present')

    opt, _ = parser.parse_known_args()

    data_dir = opt.data_dir #"data/male_female/"
    batch_size = opt.batchSize#2
    epochs = opt.epochs#40
    lambda_cyc = opt.lambda_cyc#10
    lambda_idt = opt.lambda_idt#5
    save_epoch_freq = opt.save_epoch_freq#5
    num_resnet_blocks = opt.num_resnet_blocks#9

    trainA, trainB = helpers.load_train_images(data_dir)
    train_optimizer = Adam(0.0002, 0.5)

    #Define the two discriminator models
    discA = networks.define_discriminator_network()
    discB = networks.define_discriminator_network()

    print(discA.summary())

    #The discriminators are trained on MSE loss on the patch output
    #Compile the model for dicriminators
    discA.compile(loss='mse', optimizer=train_optimizer, metrics= ['accuracy'])
    discB.compile(loss='mse', optimizer=train_optimizer, metrics= ['accuracy'])

    real_labels = np.ones((batch_size, 7, 7, 1))
    fake_labels = np.zeros((batch_size, 7, 7, 1))

    #Define the two generator models
    genA2B = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)
    genB2A = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)

    print(genA2B.summary())    

    #make the dicriminators non-trainable in the adversarial model
    discA.trainable = False
    discB.trainable = False

    #Define the adversarial model
    gan_model = networks.define_adversarial_model(genA2B, genB2A, discA, discB, train_optimizer, lambda_cyc=lambda_cyc, lambda_idt=lambda_idt)

    #Setup the tensorboard to store and visualise losses
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True,
                                  write_graph=True)
    tensorboard.set_model(genA2B)
    tensorboard.set_model(genB2A)
    tensorboard.set_model(discA)
    tensorboard.set_model(discB)
    print("Batch Size: {}".format(batch_size))
    print("Num of ResNet Blocks: {}".format(num_resnet_blocks))
    print("Starting training for {0} epochs with lambda_cyc = {1}, lambda_idt = {2}, num_resnet_blocks = {3}".format(epochs, lambda_cyc, lambda_idt, num_resnet_blocks))
    #Start training
    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))
        start_time = time.time()

        dis_losses = []
        gen_losses = []

        num_batches = int(min(trainA.shape[0], trainB.shape[0]) / batch_size)
        print("Number of batches:{} in each epoch".format(num_batches))

        for index in range(num_batches):
            print("Batch:{}".format(index))

            # Sample images
            realA = trainA[index * batch_size:(index + 1) * batch_size]
            realB = trainB[index * batch_size:(index + 1) * batch_size]

            # Translate images to opposite domain
            fakeB = genA2B.predict(realA)
            fakeA = genB2A.predict(realB)

            # Train the discriminator A on real and fake images
            dLossA_real = discA.train_on_batch(realA, real_labels)
            dLossA_fake = discA.train_on_batch(fakeA, fake_labels)

            # Train the discriminator B on ral and fake images
            dLossB_real = discB.train_on_batch(realB, real_labels)
            dLossB_fake = discB.train_on_batch(fakeB, fake_labels)

            # Calculate the total discriminator loss
            mean_disc_loss = 0.5 * np.add(0.5 * np.add(dLossA_real, dLossA_fake), 0.5 * np.add(dLossB_real, dLossB_fake))

            print("Total Discriminator Loss:{}".format(mean_disc_loss))

            """
            Train the generator networks
            """
            g_loss = gan_model.train_on_batch([realA, realB],
                                                        [real_labels, real_labels, realA, realB, realA, realB])

            print("Adversarial Model losses:{}".format(g_loss))

            dis_losses.append(mean_disc_loss)
            gen_losses.append(g_loss)

        #Save losses to tensorboard for that epoch
        #Adding a smoothed out loss (by taking mean) to the tensorboard
        helpers.save_losses_tensorboard(tensorboard, 'discriminatorA_loss', np.mean(0.5 * np.add(dLossA_real, dLossA_fake)), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'discriminatorB_loss', np.mean(0.5 * np.add(dLossB_real, dLossB_fake)), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
        if epoch % save_epoch_freq == 0:
            # Load Test images for seeing the results of the network
            testA, testB = helpers.load_test_images(data_dir=data_dir, num_images=2)

            # Generate images
            fakeB = genA2B.predict(testA)
            fakeA = genB2A.predict(testB)

            # Get reconstructed images
            reconsA = genB2A.predict(fakeB)
            reconsB = genA2B.predict(fakeA)

            identityA = genB2A.predict(testA)
            identityB = genA2B.predict(testB)

            genA2B.save('generatorAToB_temp_%d.h5'%epoch)
            genB2A.save('generatorBToA_temp_%d.h5'%epoch)
            discA.save('discriminatorA_temp_%d.h5'%epoch)
            discB.save('discriminatorB_temp_%d.h5'%epoch)
            helpers.save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsA, identityA, identityB)

        print("--- %s seconds --- for epoch" % (time.time() - start_time))

    print("Training completed. Saving weights.")
    genA2B.save('generatorAToB.h5')
    genB2A.save('generatorBToA.h5')
    discA.save('discriminatorA.h5')
    discB.save('discriminatorB.h5')


    
