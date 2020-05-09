import os
import re
import random
import argparse
import numpy as np
import tensorflow as tf


def getFiles(directory):
    """
    Given a directory, extract every jpg file in the directory and its
    subdirectories and store in a list to return
    """
    # Create a list of files and sub-directories in current directory
    directory_items = os.listdir(directory)
    directory_jpgs = []

    # Iterate over all the directory contents
    for item in directory_items:
        # Create full path for current directory
        fullPath = os.path.join(directory, item)

        # If entry is a directory then get list of files in this directory
        if os.path.isdir(fullPath):
            directory_jpgs = directory_jpgs + getFiles(fullPath)
        elif fullPath.endswith(".jpg"):
            directory_jpgs.append(fullPath)

    # Return list of jpg files in directory
    return directory_jpgs


def getPairs(filenames):
    """
    Given a list of filenames, match each filename with another filename at
    random, and return list of pairings
    """
    paired_filenames = []
    dog_breeds = []

    for file_ in filenames:
        # Get breed of dog from filename
        file_breed = re.compile(r"(?<=\-)(.*?)(?=\/)").split(file_)[1]
        if file_breed not in dog_breeds:
            dog_breeds.append(file_breed)

        # Declare condition for breaking out of while loop
        same_breed = True

        # While loop to pair a filename with another type of dog breed
        while same_breed:
            # Randomly select another filename
            file_pair = random.sample(filenames, 1)[0]

            # Get breed of dog for randomly selected filename
            file_pair_breed = re.compile(r"(?<=\-)(.*?)(?=\/)").split(file_pair)[1]

            # Check whether breeds are the same or not
            if file_breed != file_pair_breed:
                same_breed = False
                paired_filenames.append([file_, file_pair])

    # Return the list of filename pairs and dog breeds
    return paired_filenames, dog_breeds

def getDataset(file_pairs, img_height, img_width, batch_size=100, shuffle_buffer_size=20580, n_threads=2):
    """
    Given a list of file pairs and a batch size, the following method returns a
    dataset iterator that can be queried for a batch of images

    :param: file_pairs: list of file pairs
    :param: batch_size: batch size of images that will be trained on every time
    :param: shuffle_buffer_size: number of elements from this dataset that the
                                 new dataset will sample from (20,580 total images)
    :param: n_thread: number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
    # Function used to load and pre-process image files
    def getImage(file_pair):
        """
        Given a pair of file paths, open the images stored in the file
        and extract their relevant dog breed caption

        :param: file_pair: a pair of file paths

        :return: rgb image, caption, rgb image, caption
        """
        # Get filepaths
        corr_filepath = file_pair[0]
        wrong_filepath = file_pair[1]

        # corr_string_filepath = corr_filepath.numpy().decode("utf-8")
        # wrong_string_filepath = wrong_filepath.numpy().decode("utf-8")

        # Load images from filepaths
        corr_image = tf.io.decode_jpeg(tf.io.read_file(corr_filepath), channels=3)
        wrong_image = tf.io.decode_jpeg(tf.io.read_file(wrong_filepath), channels=3)

        # Convert image to normalized float (0, 1)
        corr_image = tf.image.convert_image_dtype(corr_image, tf.float32)
        wrong_image = tf.image.convert_image_dtype(wrong_image, tf.float32)

        # Resize image to img_height x img_width
        corr_image = tf.image.resize(corr_image, [img_height, img_width])
        wrong_image = tf.image.resize(wrong_image, [img_height, img_width])

        # # Determine captions of the correct and wrong images
        # corr_breed = re.compile(r"(?<=\-)(.*?)(?=\/)").split(corr_string_filepath)[1]
        # wrong_breed = re.compile(r"(?<=\-)(.*?)(?=\/)").split(wrong_string_filepath)[1]

        ### Not sure if rescaling is needed (revisit this)
        # Rescale image to range (-1, 1)
        corr_image = (corr_image - 0.5) * 2
        wrong_image = (wrong_image - 0.5) * 2

        # Return processed image
        # return corr_image, corr_breed, wrong_image, wrong_breed corr_image, corr_filepath, wrong_image, wrong_filepath]
        return corr_image, corr_filepath, wrong_image, wrong_filepath

    # Store filenames_pairs into tf dataset
    dataset = tf.data.Dataset.from_tensor_slices(file_pairs)

    # Shuffle order of dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=getImage, num_parallel_calls=n_threads)

    # Create batch, drop the final one which has less than batch_size elements,
    # and finally set to reshuffle the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset