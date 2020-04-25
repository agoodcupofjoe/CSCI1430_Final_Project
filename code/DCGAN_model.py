# Library Imports
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_gan as tfgan

# Library Function Imports
from imageio import imwrite
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

# Written File/Function Imports
from preprocess import load_image_batch

# Killing optional CPU driver warnings/Prevent showing information/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Determine GPU Status
gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## Command Line Arguments
## -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DCGAN')

# TODO: Change default to name of dog directory
parser.add_argument('--img-dir', type=str, default='../data/dogs',
                    help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='../output',
                    help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

# TODO: Reevaluate training parameters (i.e. batch size, epoch size,...etc.)
parser.add_argument('--z-dim', type=int, default=100,
                    help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
                    help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
                    help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=7,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()

## --------------------------------------------------------------------------------------

# Numerically stable logarithm function
def log(x):
    """
    Finds the stable log of x

    :param x:
    """
    return tf.math.log(tf.maximum(x, 1e-5))

## --------------------------------------------------------------------------------------

# Function for calculating Frechet Inception Distance of the generated images
# Courtesy of: CSCI 1470 - Deep Learning at Brown University
module = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])
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
    INCEPTION_IMAGE_SIZE = (299, 299)
    real_resized = tf.image.resize(real_image_batch, INCEPTION_IMAGE_SIZE)
    fake_resized = tf.image.resize(generated_image_batch, INCEPTION_IMAGE_SIZE)
    module.build([None, 299, 299, 3])
    real_features = module(real_resized)
    fake_features = module(fake_resized)
    return tfgan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)

class Generator_Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()

        # Optimizer definition:
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learn_rate, beta_1=args.beta1)

        # Loss definition:
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # Dense layer that takes in text feature vector:
        self.generator_dense = Dense(16384)
        self.generator_reshape = Reshape((4, 4, 1024))
        self.generator_bnorm = BatchNormalization()

        # First convolution layer and batch normalization:
        self.generator_conv_1 = Conv2DTranspose(
            filters=512,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.gbatch_norm_1 = BatchNormalization()

        # Second convolution layer and batch normalization:
        self.generator_conv_2 = Conv2DTranspose(
            filters=256,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.gbatch_norm_2 = BatchNormalization()

        # Third convolution layer and batch normalization:
        self.generator_conv_3 = Conv2DTranspose(
            filters=128,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.gbatch_norm_3 = BatchNormalization()

        # Fourth convolution layer and batch normalization:
        self.generator_conv_4 = Conv2DTranspose(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.gbatch_norm_4 = BatchNormalization()

        # Final convolution layer for synthesis image:
        # Output Image Shape: (128 x 128 x 3) or (64 x 64 x 3), depending on
        # whether we use the fourth convolution layer or not
        self.final_conv = Conv2DTranspose(
            filters=3,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)

    @tf.function
    def call(self, inputs):
        """
        ::Input(s)::
        - @inputs: batch of text vectors; [batch_size, z_dim]


        ::Objective::
        - Executes the generator model on the feature vector from text
        descriptor and produces synthesized image

        ::return::
        - prescaled batch of generated images; [batch_size, height, width, channel]
        """
        # Pass the batch of input vectors through a dense layer
        # Batch Size x Z Dimension
        densed_inputs = self.generator_dense(inputs)

        # Reshape the outputs of the dense layer and relu the output
        # Batch Size x 4 x 4 x 1024
        reshaped_inputs = self.generator_reshape(densed_inputs)
        reshaped_inputs = self.generator_bnorm(reshaped_inputs)

        # Pass the reshaped inputs through the first convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 8 x 8 x 512
        first_conv = self.generator_conv_1(reshaped_inputs)
        first_bnorm = self.gbatch_norm_1(first_conv)
        first_relu = tf.nn.relu(first_bnorm)

        # Pass the first layer outputs through the second convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 16 x 16 x 256
        second_conv = self.generator_conv_2(first_relu)
        second_bnorm = self.gbatch_norm_2(second_conv)
        second_relu = tf.nn.relu(second_bnorm)

        # Pass the second layer outputs through the third convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 32 x 32 x 128
        third_conv = self.generator_conv_3(second_relu)
        third_bnorm = self.gbatch_norm_3(third_conv)
        third_relu = tf.nn.relu(third_bnorm)

        ## TODO: Decide whether we use this fourth convolution layer or not
        ## Maybe have to remove it if architecture is too complex

        # Pass the third layer outputs through the fourth convolution layer
        # Includes: Conv2dTranspose, BatchNormalization, ReLU
        # Batch Size x 64 x 64 x 64
        fourth_conv = self.generator_conv_3(third_relu)
        fourth_bnorm = self.gbatch_norm_3(fourth_conv)
        fourth_relu = tf.nn.relu(fourth_bnorm)

        # Pass the fourth layer outputs through the final convolution layer
        # Batch Size x 128 x 128 x 3
        final_conv = self.final_conv(fourth_relu)

        # Tanh the final layer output
        return tf.nn.tanh(final_conv)

    @tf.function
    def loss_function(self, disc_fake_output):
        """
        ::Input(s)::
        - disc_fake_output:
            discrimator output for the generated images; [batch_size,1]

        ::Objective::
        - Outputs the loss given the discriminator output on the generated images.

        :return: loss, the cross entropy loss, scalar
        """
        # Determine binary cross entropy loss using all ones for the labels
        # and the discriminator outputs for the predictions
        return self.loss(tf.ones_like(disc_fake_output), disc_fake_output)

class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the discriminator network is defined here.
        """
        super(Discriminator_Model, self).__init__()

        # Optimizer definition:
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=(args.learn_rate / 3), beta_1=args.beta1)

        # Loss definition:
        self.real_loss = tf.keras.losses.BinaryCrossentropy()
        self.fake_loss = tf.keras.losses.BinaryCrossentropy()

        # Model definition:
        # First convolution layer and leaky ReLU
        self.discriminator_conv_1 = Conv2D(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.leaky_relu_1 = LeakyReLU(alpha=0.2)

        # Second convolution layer, batch normalization, and leaky ReLU
        self.discriminator_conv_2 = Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.dbatch_norm_2 = BatchNormalization()
        self.leaky_relu_2 = LeakyReLU(alpha=0.2)

        # Third convolution layer, batch normalization, and leaky ReLU
        self.discriminator_conv_3 = Conv2D(
            filters=256,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.dbatch_norm_3 = BatchNormalization()
        self.leaky_relu_3 = LeakyReLU(alpha=0.2)

        # Fourth convolution layer, batch normalization, and leaky ReLU
        self.discriminator_conv_4 = Conv2D(
            filters=512,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.dbatch_norm_4 = BatchNormalization()
        self.leaky_relu_4 = LeakyReLU(alpha=0.2)

        # Final convolution layer, batch normalization, and leaky ReLU
        self.discriminator_conv_5 = Conv2D(
            filters=1024,
            kernel_size=(5,5),
            strides=(2,2),
            padding="SAME",
            use_bias=False)
        self.dbatch_norm_4 = BatchNormalization()
        self.leaky_relu_4 = LeakyReLU(alpha=0.2)

        # Layer for flattening the generated outputs and a dense layer
        # for generating a prediction
        self.discriminator_flatten = Flatten()
        self.discriminator_dense = Dense(1, activation="sigmoid")

    @tf.function
    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        # Pass the inputs through the first convolution layer
        # Includes: Conv2d and leaky ReLU
        # Output: Batch size x 64 x 64 x 64
        first_conv = self.discriminator_conv_1(inputs)
        first_lrelu = self.leaky_relu_1(first_conv)

        # Pass the first layer outputs through the second convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 32 x 32 x 128
        second_conv = self.discriminator_conv_2(first_lrelu)
        second_bnorm = self.dbatch_norm_2(second_conv)
        second_lrelu = self.leaky_relu_2(second_bnorm)

        # Pass the second layer outputs through the third convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 16 x 16 x 256
        third_conv = self.discriminator_conv_3(second_lrelu)
        third_bnorm = self.dbatch_norm_3(third_conv)
        third_lrelu = self.leaky_relu_3(third_bnorm)

        # Pass the third layer outputs through the fourth convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 8 x 8 x 512
        fourth_conv = self.discriminator_conv_4(third_lrelu)
        fourth_bnorm = self.dbatch_norm_4(fourth_conv)
        fourth_lrelu = self.leaky_relu_4(fourth_bnorm)

        # Pass the fourth layer outputs through the final convolution layer
        # Includes: Conv2d, BatchNormalization, and leaky ReLU
        # Output: Batch size x 4 x 4 x 1024
        final_conv = self.discriminator_conv_4(fourth_lrelu)
        final_bnorm = self.dbatch_norm_4(final_conv)
        final_lrelu = self.leaky_relu_4(final_bnorm)

        # Flatten the output of the final convolution layer
        flattened_output = self.discriminator_flatten(final_lrelu)

        # Return the output of the dense layer with sigmoid activation
        return self.discriminator_dense(flattened_output)

    def loss_function(self, disc_real_output, disc_fake_output):
        """
        Outputs the discriminator loss given the discriminator model output on the real and generated images.

        :param disc_real_output: discriminator output on the real images, shape=[batch_size, 1]
        :param disc_fake_output: discriminator output on the generated images, shape=[batch_size, 1]

        :return: loss, the combined cross entropy loss, scalar
        """
        # Determine binary cross entropy loss using all ones for the labels
        # and the discriminator outputs for the batch of real images
        real_loss = self.real_loss(tf.ones_like(disc_real_output), disc_real_output)

        # Determine binary cross entropy loss using all zeros for the labels
        # and the discriminator outputs for the batch of generated images
        fake_loss =  self.fake_loss(tf.zeros_like(disc_fake_output), disc_fake_output)

        # Return the sum of real_loss and fake_loss
        return (real_loss + fake_loss)

## --------------------------------------------------------------------------------------

# Train the model for one epoch.
def train(generator, discriminator, dataset_iterator, manager):
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

    # Loop over the dataset of images
    for iteration, batch in enumerate(dataset_iterator):
        # Create a batch of noise vectors
        noise = tf.random.uniform([args.batch_size, args.z_dim], minval=-1, maxval=1)

        # Set the gradient tapes for determining gradients
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # Generate fake images using the generator model and the noise vectors
            gen_output = generator.call(noise)

            # Use discriminator to generate predictions on batch of real images
            real_discriminator = discriminator.call(batch)

            # Use discriminator to generate predictions on batch of fake images
            fake_discriminator = discriminator.call(gen_output)

            # Determine loss of generator and discriminator model
            generator_loss = generator.loss_function(fake_discriminator)
            discriminator_loss = discriminator.loss_function(real_discriminator, fake_discriminator)

        # Determine the gradients of generator and discriminator model
        generator_gradient = generator_tape.gradient(
            generator_loss, generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(
            discriminator_loss, discriminator.trainable_variables)

        # Use the optimizers to apply the gradients to their respective models
        generator.optimizer.apply_gradients(
            zip(generator_gradient, generator.trainable_variables))
        discriminator.optimizer.apply_gradients(
            zip(discriminator_gradient, discriminator.trainable_variables))

        # Save
        if iteration % args.save_every == 0:
            manager.save()

        # Calculate inception distance and track the fid to return the average
        if iteration % 500 == 0:
            fid_ = fid_function(batch, gen_output)
            fid_tracker += fid_
            tracker += 1
            print('**** INCEPTION DISTANCE: %g ****' % fid_)

    # Return the average fid based off of the fids printed to the console
    return (fid_tracker / tracker)

# Test the model by generating some samples.
def test(generator):
    """
    Test the model.

    :param generator: generator model

    :return: None
    """
    # Sample and generate a batch of random images
    noise = tf.random.uniform([args.batch_size, args.z_dim], minval=-1, maxval=1)
    img = generator.call(noise)
    img = img.numpy()

    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255

    # Convert to uint8
    img = img.astype(np.uint8)

    # Save images to disk
    for i in range(0, args.batch_size):
        img_i = img[i]
        s = args.out_dir+'/'+str(i)+'.png'
        imwrite(s, img_i)

## --------------------------------------------------------------------------------------

def main():
    # Load a batch of images (to feed to the discriminator)
    dataset_iterator = load_image_batch(args.img_dir, batch_size=args.batch_size, n_threads=args.num_data_threads)

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

    if args.restore_checkpoint or args.mode == 'test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint)

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    avg_fid = train(generator, discriminator, dataset_iterator, manager)
                    print("Average FID for Epoch: " + str(avg_fid))
                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
            if args.mode == 'test':
                test(generator)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()


