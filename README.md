# CSCI 1430, Spring 2020 - Final Project
Text-to-Image Synthesis of Dogs via DCGAN\
By. Esther Kim, Joseph Chen, Justin Zhang

For our project submission, we have included only one training data file and one output file as examples, due to concerns regarding the size of our submission in Gradescope.

Before running the code, it will be necessary to create a virtual environment, with the venv_requirements.txt file provided in the code directory. In addition to creating a virtual environment with the packages listed in the venv_requirements.txt file, one must also run the following commands:
pip3 install tensorflow_hub
pip3 install tensorflow_gan

Finally, to run our code, the only files required are "DCGAN_preprocess.py" and "DCGAN_train.py".
1. Make sure that both "DCGAN_preprocess.py" and "DCGAN_train.py" are located in the code directory
2. Make sure that training data can be located in a directory with the following file path (relative to the code directory):
   "../data/Images/"
3. To begin training, run the following command from the code directory:
   "python DCGAN_train.py"
4. To test the generator and synthesize images, run the following command from the code directory:
   "python DCGAN_train.py --test"
   
*Our code utilizes argparse so that command line parameters can be incorporated into the training and tune hyperparameters. For more information on how to specify certain command line arguments, look into the argparse section in "DCGAN_train.py"

Cheers!
