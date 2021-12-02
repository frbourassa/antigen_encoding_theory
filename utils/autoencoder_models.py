""" Simple autoencoder models with the Keras model subclassing API, 
and functions to save and load those custom classes. 

The classes are based on the Tensorflow autoencoder tutorial:
https://www.tensorflow.org/tutorials/generative/autoencoder

I added methods to save and load. It turns out using Sequential
in a subclass of Model like in the tutorial was not a good choice, 
because all save and load functions needed to be redefined
to go get the layers within the Sequential models. 

@author: frbourassa
November 5, 2021
"""
from tensorflow import keras as ks
import os, json

from tensorflow.python.platform import tf_logging
import tensorflow.python.keras.saving.hdf5_format as ks_saving
try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None



# sys to get the different classes defined in this module with getattr
# and pick the correct one when loading an autoencoder model
# See https://stackoverflow.com/questions/2933470/how-do-i-call-setattr-on-the-current-module
import sys

# Keras subclassing API
class Autoencoder(ks.models.Model):
    def __init__(self, inputdim, latentdim, hidden_activ="tanh", out_activ="tanh"):
        super(Autoencoder, self).__init__()
        # Save all parameters as attributes to allow model saving and re-creation
        self.latent_dim = latentdim
        self.input_dim = inputdim
        self.hidden_activ = hidden_activ
        self.out_activ = out_activ
        
        # Create layers
        self.encoder = ks.Sequential([
            ks.layers.Input(inputdim),
            ks.layers.Dense(latentdim, activation=hidden_activ)
        ])
        self.decoder = ks.layers.Dense(inputdim, activation=out_activ)
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def to_json(self):
        """ Saves a dictionary of parameters that allows to
        recreate a new Autoencoder_regul with the same architecture.
        Returns a string giving the JSON file contents. """
        params = {
            "class": type(self).__name__, 
            "inputdim": self.input_dim, 
            "latentdim": self.latent_dim, 
            "hidden_activ": self.hidden_activ, 
            "out_activ": self.out_activ,
        }
        return json.JSONEncoder().encode(params)
    
    def check_weights_length(self, model):
        """ For either the encoder or the decoder """
        if len(model.weights) != len(model._undeduplicated_weights):
            tf_logging.warning('Found duplicated `Variable`s in Model\'s `weights`. '
                    'This is usually caused by `Variable`s being shared by '
                    'Layers in the Model. These `Variable`s will be treated '
                    'as separate `Variable`s when the Model is restored. To '
                    'avoid this, please save with `save_format="tf"`.')
        
    def save_weights(self, filepath, overwrite=False):
        """ Saved weights of all layers """
        # Using the keras source code to create the h5py
        if h5py is None:
            raise ImportError('`save_model` requires h5py.')

        #self.check_weights_length(self.encoder)
        #self.check_weights_length(self.decoder)
        if not isinstance(filepath, h5py.File):
            # If file exists, it should not be overwritten. 
            if not overwrite and os.path.isfile(filepath):
                proceed = False  # ask_to_proceed_with_overwrite(filepath)
                print("Could not save: file {} already exists.".format(filepath))
                if not proceed:
                    return None

            f = h5py.File(filepath, mode='w')
            opened_new_file = True
        else:
            f = filepath
            opened_new_file = False
        
        # Create one group for the encoder layer
        encoder_group = f.create_group('encoder_weights')
        
        # Use the keras saving internal function for each layer in the encoder
        ks_saving.save_weights_to_hdf5_group(encoder_group, self.encoder.layers)
        
        # Create one group for the decoder layer. 
        decoder_group = f.create_group('decoder_weights')
        ks_saving.save_weights_to_hdf5_group(decoder_group, [self.decoder])
        
        f.flush()
        if opened_new_file:
            f.close()
        
        return 0
        
    
    def load_weights(self, filepath):
        """ Put back loaded weights in the layers """
        if h5py is None:
            raise ImportError('`load_model` requires h5py.')

        opened_new_file = not isinstance(filepath, h5py.File)
        if opened_new_file:
            f = h5py.File(filepath, mode='r')
        else:
            f = filepath
        
        # Copying and simplifying keras source code to deal with file creating
        # Use the keras saving internal function to put back weights of each group
        ks_saving.load_weights_from_hdf5_group(f['encoder_weights'], self.encoder.layers)
        # Decoder is a single layer; need to build it to initialize weights object
        # before loading the saved weight values into that object
        # This was not necessary for the encoder because the Sequential Model
        # already builds the layer weights. 
        self.decoder.build(self.latent_dim)
        ks_saving.load_weights_from_hdf5_group(f['decoder_weights'], [self.decoder])
        
        if opened_new_file:
              f.close()
        
        return self
        
        
    
# With regularization on the hidden layer
class Autoencoder_regul(Autoencoder):
    def __init__(self, inputdim, latentdim, hidden_activ="tanh", out_activ="tanh", regrate=1e-4):
        super(Autoencoder, self).__init__()  # parent of the basic Autoencoder class: Model
        # Save all parameters as attributes to allow model saving and re-creation
        self.latent_dim = latentdim
        self.input_dim = inputdim
        self.reg_rate = regrate
        self.hidden_activ = hidden_activ
        self.out_activ = out_activ
        
        # Create layers
        self.encoder = ks.Sequential([
            ks.layers.Input(inputdim),
            ks.layers.Dense(latentdim, activation=hidden_activ, 
                            activity_regularizer=ks.regularizers.l1(regrate))
        ])
        self.decoder = ks.layers.Dense(inputdim, activation=out_activ)
    
    # def call(self, x):  # Inherited from Autoencoder

    
    def to_json(self):
        """ Saves a dictionary of parameters that allows to
        recreate a new Autoencoder_regul with the same architecture. 
        Returns a string giving the JSON file contents. """
        params = {
            "class": type(self).__name__, 
            "inputdim": self.input_dim, 
            "latentdim": self.latent_dim, 
            "hidden_activ": self.hidden_activ, 
            "out_activ": self.out_activ, 
            "regrate": self.reg_rate
        }
        return json.JSONEncoder().encode(params)
    
    #def save_weights(self, filepath, overwrite=False):  # Inherited from Autoencoder
    #def load_weights(self, filepath):  # Inherited from Autoencoder
        

## Functions to save and load autoencoders. 
# Define functions that properly wrap the autoencoder's name
# into a full path, so it is always uniform when loading/saving. 
def archi_file(name, folder=""):
    return os.path.join(folder, "{}_architecture.json".format(name))


def weights_file(name, folder=""):
    return os.path.join(folder, "{}_weights.h5".format(name))


def autoencoder_from_json(jfile):
    with open(jfile, "r") as handle:
        params = json.load(handle)
    
    # First, get the correct model type
    thismodule = sys.modules[__name__]
    model_class = getattr(thismodule, params.pop("class", "Autoencoder"), Autoencoder)
    
    # Generate a model with the parameters in the json file
    # The class param was popped so it won't be passed as an init argument
    # Other parameters should all be accepted
    auto = model_class(**params)
    return auto


def save_autoencoder(auto, name, folder="", overwrite=False):
    """ Utility function to save an autoencoder in a single command. 
    Architecture is saved in json and weights, in h5. 
    name does not contain the extension; it's the name of the autoencoder. 
    """
    # Saving model architecture
    with open(archi_file(name, folder), 'w') as handle:
        handle.write(auto.to_json())

    # Saving model weights
    auto.save_weights(weights_file(name, folder), overwrite=overwrite)
    return 0

    
def load_autoencoder(name, folder):
    """ Utility function to save an autoencoder in a single command. 
    Architecture is loaded from json and weights, in h5. 
    name does not contain the extension; it's the name of the autoencoder.
    """
    # Import architecture and create blank model from it
    auto = autoencoder_from_json(archi_file(name, folder))

    # Add weights to this model
    auto.load_weights(weights_file(name, folder))
    return auto
    