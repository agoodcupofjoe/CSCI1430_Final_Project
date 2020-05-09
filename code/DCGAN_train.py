### Import packages and functions
import os
import re
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_gan as tfgan
from tensorflow.keras import Model, backend, activations
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Concatenate, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda


### Import written functions
from DCGAN_preprocess import getFiles, getPairs, getDataset


### Preliminary Setup ##########################################################
## -------------------------------------------------------------------------- ##
# Killing optional CPU driver warnings/Prevent showing information/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Determine GPU Status
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
## -------------------------------------------------------------------------- ##
### End of Preliminary Setup ###################################################


### Command Line Arguments #####################################################
## -------------------------------------------------------------------------- ##
parser = argparse.ArgumentParser(description='DCGAN')

# Args for directories
parser.add_argument('--img-dir', type=str, default='../data/Images/',
                    help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='../output',
                    help='Data where sampled output images will be written to')

# Args for dimensions
parser.add_argument('--img-height', type=int, default=128,
                    help='Height of images fed through the network')

parser.add_argument('--img-width', type=int, default=128,
                    help='Width of images fed through the network')

# 05-04-2020: Changed z-dim to 128
parser.add_argument('--z-dim', type=int, default=128,
                    help='Dimensionality of the latent space')

# Args for training
parser.add_argument('--num-epochs', type=int, default=15,
                    help='Number of passes to make through the training data')

parser.add_argument('--num-batches', type=int, default=100,
                    help='Number of images in a batch to be fed through the network')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--num-gen-updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')

# Args for optimizer
parser.add_argument('--learn-rate', type=float, default=0.0002,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
                    help='"beta1" parameter for Adam optimizer')

parser.add_argument('--beta2', type=float, default=0.9,
                    help='"beta2" parameter for Adam optimizer')

# Args for checkpoints
parser.add_argument('--save-every', type=int, default=100,
                    help='Save state of the network after every [this many] training iterations')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Flag for resuming training from a previously-saved checkpoint')

# Args for GCP
parser.add_argument('--device', type=str, default='GPU:0' if len(physical_devices) > 0 else 'CPU:0',
                    help='specify the device of computation eg. CPU:0, GPU:0, ... ')

args = parser.parse_args()
## -------------------------------------------------------------------------- ##
### End of Command Line Arguments ##############################################


### FID Calculation Function ###################################################
## -------------------------------------------------------------------------- ##
# Function for calculating Frechet Inception Distance of the generated images
# Courtesy of: CSCI 1470 - Deep Learning at Brown University

def fid_function(real_image_batch, generated_image_batch):
    """
    ::Input(s)::
    - @real_image_batch:
        batch of real images; [batch_size, height, width, channels]
    - @generated_image_batch:
        batch of generated images; [batch_size, height, width, channels]

    ::Objective::
    - Pulls a pre-trained inception v3 network and uses it to extract
    activations for both the real and generated images. Then determines the
    distance of the activations.

    ::return::
    - Frechet inception distance between the real and generated images
    """
    module = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])
    INCEPTION_IMAGE_SIZE = (299, 299)
    real_resized = tf.image.resize(real_image_batch, INCEPTION_IMAGE_SIZE)
    fake_resized = tf.image.resize(generated_image_batch, INCEPTION_IMAGE_SIZE)
    module.build([None, 299, 299, 3])
    real_features = module(real_resized)
    fake_features = module(fake_resized)
    return tfgan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)
## -------------------------------------------------------------------------- ##
### End of FID Calculation Function ############################################


### Generator Model ############################################################
## -------------------------------------------------------------------------- ##
class Generator_Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()

        # Optimizer definition:
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learn_rate, beta_1=args.beta1, beta_2=args.beta2)

        # Loss function definition:
        self.reduce_mean1 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy1 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))
        self.reduce_mean2 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy2 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))

        # Dense layer for text embedding:
        # TODO: Consider changing the ouput size of the Dense layer
        self.txt_dense = Dense(128)
        # self.txt_dense = Dense(64)
        self.txt_leakyrelu = LeakyReLU(alpha=0.2)
        self.txt_concatenate = Concatenate(axis=1)

        # Dense layer for noise and text embedding:
        self.g_dense = Dense(16384)
        self.g_reshape = Reshape((4, 4, 1024))
        self.g_bnorm = BatchNormalization()

        # First convolution layer and batch normalization:
        self.g_conv1 = Conv2DTranspose(
            filters=512,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.g_bnorm1 = BatchNormalization()

        # Second convolution layer and batch normalization:
        self.g_conv2 = Conv2DTranspose(
            filters=256,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.g_bnorm2 = BatchNormalization()

        # Third convolution layer and batch normalization:
        self.g_conv3 = Conv2DTranspose(
            filters=128,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.g_bnorm3 = BatchNormalization()

        # Fourth convolution layer and batch normalization:
        self.g_conv4 = Conv2DTranspose(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.g_bnorm4 = BatchNormalization()

        # Final convolution layer for synthesis image:
        # Output Image Shape: (128 x 128 x 3)
        self.g_conv_final = Conv2DTranspose(
            filters=3,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)

        self.tanh = Lambda(activations.tanh)

    @tf.function
    def call(self, noise, text):
        """
        ::Input(s)::
        - @inputs: batch of text vectors; [batch_size, z_dim]

        ::Objective::
        - Executes the generator model on the feature vector from text
        descriptor and produces synthesized image

        ::return::
        - prescaled batch of generated images; [batch_size, height, width, channel]
        """
        # Pass text through Dense and LeakyRelu layers
        text_dense = self.txt_leakyrelu(self.txt_dense(text))

        # Concatenate noise and processed text
        dense_input = self.txt_concatenate([noise, text_dense])

        # Pass the batch of input vectors through a dense layer
        # Batch Size x 16384 Dimension
        densed_input = self.g_dense(dense_input)

        # Reshape the outputs of the dense layer and relu the output
        # Batch Size x 4 x 4 x 1024
        conv_input = self.g_reshape(densed_input)
        conv_input = self.g_bnorm(conv_input)

        # Pass the reshaped inputs through the first convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 8 x 8 x 512
        out_1 = self.g_conv1(conv_input)
        out_1 = self.g_bnorm1(out_1)
        out_1 = tf.nn.relu(out_1)

        # Pass the first layer outputs through the second convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 16 x 16 x 256
        out_2 = self.g_conv2(out_1)
        out_2 = self.g_bnorm2(out_2)
        out_2 = tf.nn.relu(out_2)

        # Pass the second layer outputs through the third convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 32 x 32 x 128
        out_3 = self.g_conv3(out_2)
        out_3 = self.g_bnorm3(out_3)
        out_3 = tf.nn.relu(out_3)

        # Pass the third layer outputs through the fourth convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 64 x 64 x 64
        out_4 = self.g_conv4(out_3)
        out_4 = self.g_bnorm4(out_4)
        out_4 = tf.nn.relu(out_4)

        # Pass the fourth layer outputs through the final convolution layer
        # Batch Size x 128 x 128 x 3
        out_final = self.g_conv_final(out_4)

        # Tanh the final layer output
        return self.tanh(out_final)

    @tf.function
    def loss_function(self, d_logits_gcorr, d_logits_gwrong):
        """
        """
        g_loss = self.sigmoid_cross_entropy1((d_logits_gcorr, tf.ones_like(d_logits_gcorr)))
        g_loss = self.reduce_mean1(g_loss)
        g_loss += self.reduce_mean2(self.sigmoid_cross_entropy2((d_logits_gwrong, tf.ones_like(d_logits_gwrong))))
        return g_loss
## -------------------------------------------------------------------------- ##
### End of Generator Model #####################################################


### Discriminator Model ########################################################
## -------------------------------------------------------------------------- ##
class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the discriminator network is defined here.
        """
        super(Discriminator_Model, self).__init__()

        # Optimizer definition:
        # Consider: learning_rate = (args.learn_rate / 3)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learn_rate, beta_1=args.beta1, beta_2=args.beta2)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=args.learn_rate / 4, beta_1=args.beta1, beta_2=args.beta2)

        # Loss function definition:
        self.reduce_mean1 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy1 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))
        self.reduce_mean2 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy2 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))
        self.reduce_mean3 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy3 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))
        self.reduce_mean4 = Lambda(tf.reduce_mean)
        self.sigmoid_cross_entropy4 = Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0], labels=x[1]))

        # Model definition:
        # First convolution layer and leaky ReLU
        self.d_conv1 = Conv2D(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.d_lrelu1 = LeakyReLU(alpha=0.2)

        # Second convolution layer, batch normalization, and leaky ReLU
        self.d_conv2 = Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.d_bnorm2 = BatchNormalization()
        self.d_lrelu2 = LeakyReLU(alpha=0.2)

        # Third convolution layer, batch normalization, and leaky ReLU
        self.d_conv3 = Conv2D(
            filters=256,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.d_bnorm3 = BatchNormalization()
        self.d_lrelu3 = LeakyReLU(alpha=0.2)

        # Fourth convolution layer, batch normalization, and leaky ReLU
        self.d_conv4 = Conv2D(
            filters=512,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.d_bnorm4 = BatchNormalization()
        self.d_lrelu4 = LeakyReLU(alpha=0.2)

        # Dense layer for text embedding
        # TODO: Consider changing the output size of the Dense layer
        self.txt_dense = Dense(128)
        # self.txt_dense = Dense(8)
        self.txt_leakyrelu = LeakyReLU(alpha=0.2)

        # Expansion and tiling layer for text embedding
        self.txt_expand1 = Lambda(backend.expand_dims, arguments={'axis':1})
        self.txt_expand2 = Lambda(backend.expand_dims, arguments={'axis':1})
        self.txt_tile1 = Lambda(backend.tile, arguments={'n':(1, 8, 8, 1)})
        self.txt_concatenate = Concatenate(axis=3)

        # Final convolution layer, batch normalization, and leaky ReLU
        self.d_conv5 = Conv2D(
            filters=1024,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.d_bnorm5 = BatchNormalization()
        self.d_lrelu5 = LeakyReLU(alpha=0.2)

        # Layer for flattening the generated outputs and a dense layer
        # for generating a prediction
        self.d_flatten = Flatten()
        self.d_dense = Dense(1)
        self.sigmoid = Lambda(activations.sigmoid)

    @tf.function
    def call(self, image, text):
        """
        Executes the discriminator model on a batch of input images and text vectors and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        # Pass the inputs through the first convolution layer
        # Includes: Conv2d and leaky ReLU
        # Output: Batch size x 64 x 64 x 64
        out_1 = self.d_conv1(image)
        out_1 = self.d_lrelu1(out_1)

        # Pass the first layer outputs through the second convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 32 x 32 x 128
        out_2 = self.d_conv2(out_1)
        out_2 = self.d_bnorm2(out_2)
        out_2 = self.d_lrelu2(out_2)

        # Pass the second layer outputs through the third convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 16 x 16 x 256
        out_3 = self.d_conv3(out_2)
        out_3 = self.d_bnorm3(out_3)
        out_3 = self.d_lrelu3(out_3)

        # Pass the third layer outputs through the fourth convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 8 x 8 x 512
        out_4 = self.d_conv4(out_3)
        out_4 = self.d_bnorm4(out_4)
        out_4 = self.d_lrelu4(out_4)

        # Process the text vector
        text_dense = self.txt_leakyrelu(self.txt_dense(text))
        text_out = self.txt_expand1(text_dense)
        text_out = self.txt_expand2(text_out)
        text_out = self.txt_tile1(text_out)
        text_out = self.txt_concatenate([out_4, text_out])

        # Pass the fourth layer outputs through the final convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 4 x 4 x 1024
        out_final = self.d_conv5(text_out)
        out_final = self.d_bnorm5(out_final)
        out_final = self.d_lrelu5(out_final)

        # Flatten the output of the final convolution layer
        flattened_out = self.d_flatten(out_final)

        # Return the output of the dense layer with sigmoid activation
        logits = self.d_dense(flattened_out)
        outputs = self.sigmoid(logits)
        return outputs, logits

    @tf.function
    def loss_function(self, d_logits_real, d_logits_gcorr, d_logits_realwrong, d_logits_wrongcorr):
        """
        """
        d_loss1 = self.reduce_mean1(self.sigmoid_cross_entropy1((d_logits_real, tf.ones_like(d_logits_real))))
        d_loss2 = self.reduce_mean2(self.sigmoid_cross_entropy2((d_logits_gcorr, tf.zeros_like(d_logits_gcorr))))
        d_loss3 = self.reduce_mean3(self.sigmoid_cross_entropy3((d_logits_realwrong, tf.zeros_like(d_logits_realwrong))))
        d_loss4 = self.reduce_mean4(self.sigmoid_cross_entropy4((d_logits_wrongcorr, tf.zeros_like(d_logits_wrongcorr))))
        d_loss = d_loss1 + (0.5 * (d_loss2 + d_loss3 + d_loss4))
        return d_loss
## -------------------------------------------------------------------------- ##
### End of Discriminator Model #################################################


### DCGAN Train Function #######################################################
## -------------------------------------------------------------------------- ##
# Train the model for one epoch.
def train(generator, discriminator, dataset_iterator, dataset_breeds, manager):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return: The average FID score over the epoch
    """
    # Declare variables for determiniing the average FID of the epoch
    tracker = 0
    fid_tracker = 0
    gen_loss_tracker = 0
    dis_loss_tracker =  0

    # Loop over the dataset of images
    for iteration, batch in enumerate(dataset_iterator):
        def decode_bytes(filename_bytes):
            """
            Function for decoding byte variable to string variable
            """
            return filename_bytes.decode("utf-8")

        # Extract the images from the dataset_iterator
        corr_img_batch = batch[0]
        wrong_img_batch = batch[2]

        # Extract the captions from the dataset_iterator and process into embeddings
        corr_caption_batch = batch[1]
        corr_caption_batch = [decode_bytes(b) for b in corr_caption_batch.numpy()]
        corr_caption_batch = [re.compile(r"(?<=\-)(.*?)(?=\/)").split(x)[1] for x in corr_caption_batch]
        corr_caption_batch = [dataset_breeds.index(x) for x in corr_caption_batch]
        corr_caption_batch = np.array([np.eye(len(dataset_breeds), dtype=np.float32)[x] for x in corr_caption_batch])

        wrong_caption_batch = batch[3]
        wrong_caption_batch = [decode_bytes(b) for b in wrong_caption_batch.numpy()]
        wrong_caption_batch = [re.compile(r"(?<=\-)(.*?)(?=\/)").split(x)[1] for x in wrong_caption_batch]
        wrong_caption_batch = [dataset_breeds.index(x) for x in wrong_caption_batch]
        wrong_caption_batch = np.array([np.eye(len(dataset_breeds), dtype=np.float32)[x] for x in wrong_caption_batch])

        # Interpolate the batch of wrong captions
        wrong_caption_batch = (0.5 * corr_caption_batch) + (0.5 * wrong_caption_batch)

        # Convert batches of captions to tensors
        corr_caption_batch = tf.Variable(corr_caption_batch)
        wrong_caption_batch = tf.Variable(wrong_caption_batch)

        # Create a batch of noise vectors
        noise = tf.random.uniform([args.num_batches, args.z_dim], minval=-1, maxval=1)

        # Set the gradient tapes for determining gradients
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # Generate fake images using the generator model, noise, and correct text vectors
            g_imgs_corr = generator.call(noise, corr_caption_batch)

            # Generate fake images using the generator model, noise, and incorrect text vectors
            g_imgs_wrong = generator.call(noise, wrong_caption_batch)

            # Make predictions on batch of real images and correct text vectors
            d_outputs_real, d_logits_real = discriminator.call(corr_img_batch, corr_caption_batch)

            # Make predictions on batch of generated images using correct text vectors
            _, d_logits_gcorr = discriminator.call(g_imgs_corr, corr_caption_batch)

            # Make predictions on batch of generated images using incorrect text vectors
            # 05-04-2020: Changed from corr_caption_batch to wrong_caption_batch
            # 05-05-2020: Changed from wrong_caption_batch to corr_caption_batch
            _, d_logits_gwrong = discriminator.call(g_imgs_wrong, corr_caption_batch)

            # Make predictions on batch of real images and incorrect text vectors
            _, d_logits_realwrong = discriminator.call(corr_img_batch, wrong_caption_batch)

            # Make predictions on batch of wrong images and correct text vectors
            _, d_logits_wrongcorr = discriminator.call(wrong_img_batch, corr_caption_batch)

            # Determine loss of generator and discriminator model
            generator_loss = generator.loss_function(d_logits_gcorr, d_logits_gwrong)
            discriminator_loss = discriminator.loss_function(d_logits_real, d_logits_gcorr, d_logits_realwrong, d_logits_wrongcorr)

        # Determine the gradients of generator and discriminator model
        generator_gradient = generator_tape.gradient(
            generator_loss, generator.trainable_variables)

        if iteration % args.num_gen_updates == 0:
            discriminator_gradient = discriminator_tape.gradient(
                discriminator_loss, discriminator.trainable_variables)

        # Use the optimizers to apply the gradients to their respective models
        generator.optimizer.apply_gradients(
            zip(generator_gradient, generator.trainable_variables))

        if iteration % args.num_gen_updates == 0:
            discriminator.optimizer.apply_gradients(
                zip(discriminator_gradient, discriminator.trainable_variables))

        # Save
        if iteration % args.save_every == 0:
            manager.save()

        # Calculate inception distance and track the fid to return the average
        if iteration % 40 == 0:
            fid_ = fid_function(corr_img_batch, g_imgs_corr)
            fid_tracker += fid_
            gen_loss_tracker += generator_loss
            dis_loss_tracker += discriminator_loss
            tracker += 1
            print('**** INCEPTION DISTANCE: %g ****' % fid_)
            print('**** Generator Loss: %g ****' % generator_loss)
            print('**** Discriminator Loss: %g ****' % discriminator_loss)

    # Return the average fid based off of the fids printed to the console
    return (fid_tracker / tracker), (gen_loss_tracker / tracker), (dis_loss_tracker / tracker)
## -------------------------------------------------------------------------- ##
### End of DCGAN Train Function ################################################

### DCGAN Test Function ########################################################
## -------------------------------------------------------------------------- ##
# Test the model by generating some samples.
def test(generator, dataset_breeds, is_train, epoch):
    """
    """
    # Sample and generate a batch of random images
    noise = tf.random.uniform([len(dataset_breeds), args.z_dim], minval=-1, maxval=1)
    text = tf.Variable(np.eye(len(dataset_breeds), dtype=np.float32))

    img = generator.call(noise, text)
    img = img.numpy()

    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255

    # Convert to uint8
    img = img.astype(np.uint8)

    # Create directory specifically for this test function call
    # Default: '../output'
    if is_train:
        output_dir = args.out_dir +'/train16_epoch' + str(epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    elif not is_train:
        output_dir = args.out_dir +'/test0'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Save images to disk
    for i in range(0, len(dataset_breeds)):
        img_i = img[i]
        s = output_dir+'/'+str(i)+'.png'
        imwrite(s, img_i)
## -------------------------------------------------------------------------- ##
### End of DCGAN Test Function #################################################

def main():
    # # Clear the default graph stack and reset the global default graph
    # tf.compat.v1.reset_default_graph()

    # Get filenames of all the dog jpg files
    filenames = getFiles(args.img_dir)

    # Get pairs of filenames
    filenames_pairs, dataset_breeds = getPairs(filenames)
    print(dataset_breeds)

    # Get dataset of filename pairs
    dataset_iterator = getDataset(filenames_pairs, args.img_height, args.img_width, args.num_batches)

    # Initialize generator and discriminator models
    generator = Generator_Model()
    discriminator = Discriminator_Model()

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # If restoring checkpoint or testing
    if args.restore_checkpoint or args.mode == 'test':
        # Restores the latest checkpoint
        checkpoint.restore(manager.latest_checkpoint)

    # Begin training or testing
    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                # Create a file (or open if already exisiting) to record the
                # average FID score for the completed EPOCH
                # fid_output_file = open("EPOCH_FID_scores.txt", "a+")

                test(generator, dataset_breeds, True, 0)

                # fid_output_file.close()

                # Iterate through the EPOCHs
                for epoch in range(0, args.num_epochs):
                    # Begin training for the current EPOCH
                    print('========================== EPOCH %d  ==========================' % epoch)
                    # # Get new pairs of filenames
                    # print("Getting Pairs")
                    # filenames_pairs, _ = getPairs(filenames)
                    # print("Done getting pairs")

                    # # Get new dataset of filename pairs
                    # print("Getting Iterator")
                    # dataset_iterator = getDataset(filenames_pairs, args.img_height, args.img_width, args.num_batches)
                    # print("Done getting iterator")

                    # Train the generator and discriminator
                    avg_fid, avg_gloss, avg_dloss = train(generator, discriminator, dataset_iterator, dataset_breeds, manager)

                    # Print FID for the completed EPOCH
                    print("Average FID for EPOCH: " + str(avg_fid))

                    # Write the FID to the output file
                    fid_output_file = open("EPOCH_FID_scores.txt", "a+")
                    fid_output_file.write(("EPOCH %d FID: " + str(avg_fid) + "\n") % epoch)
                    fid_output_file.write(("EPOCH %d G_LOSS: " + str(avg_gloss) + "\n") % epoch)
                    fid_output_file.write(("EPOCH %d D_LOSS: " + str(avg_dloss) + "\n") % epoch)
                    fid_output_file.close()

                    # Save network state at the end of the EPOCH
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
                    print("**** SAVE COMPLETED ****")

                    # Test the generator
                    print("**** TESTING GENERATOR AT END OF EPOCH ****")
                    test(generator, dataset_breeds, True, epoch + 1)
                    print("**** TEST COMPLETED ****")


            if args.mode == 'test':
                # Run the test function on the generator with all the possible dog breeds
                test(generator, dataset_breeds, False, 0)

    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
   main()
