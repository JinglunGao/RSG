import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, activations, initializers

class ENN:
    
    """Ensemble neural networks
    Parameters
    ----------
    n_estimators: int
        number of networks
    train: list
        save the training sets len(train) == n_estimators
    test: list
        save the testing sets len(test) == n_estimators
    models: list
        save the neural networks model
    results: numpy.array
        save the training / testing results 
    
    Attributes
    ----------
    bootstrap: generate the training and testing set
    build_model: build the neural network model
    fit: fit the models
    predict: make the prediction (average)
    get_results: print out the results
    """
    
    def __init__(self, n_estimators = None, train = None, test = None, models = None, results = None):
        self.n_estimators = n_estimators
        self.train = train
        self.test = test
        self.models = models
        self.results = results
        
    def bootstrap(self, data, stratification, trainsize):
        """sampling the data with replacement
        
        Parameters
        ----------
        data : pandas.DataFrame
        stratification: numpy.array, data is split in a stratified fashion (moneyness)
        trainsize: float, percentage of data used for the training set
        """
        n = data.shape[0]
        train = []
        test = []
        idx = data.index.values
        for _ in range(self.n_estimators):
            # with replacement
            train_idx = resample(idx, n_samples = trainsize * n, 
                                 replace = True, stratify = stratification)
            test_idx = np.array([x for x in idx if x not in train_idx])
            assert (len(test_idx) + len(np.unique(train_idx))) == n, 'wrong idx'
            train.append(data.loc[train_idx])
            test.append(data.loc[test_idx])
        self.train = train
        self.test = test
        return None
    
    def build_model(self, lr):
        """Build a single neural network model
        
        Parameters
        ----------
        lr : float, learning rate
        """
        model = models.Sequential()
        # Input layer
        model.add(layers.Dense(units = 5,
                               kernel_initializer = initializers.RandomNormal(seed = 123),
                               bias_initializer = initializers.Zeros()))
        model.add(layers.Activation(activations.elu))

        numLayers = 3
        for i in range(numLayers):
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(units = 100, 
                                   kernel_initializer = initializers.RandomNormal(seed = 123),
                                   bias_initializer = initializers.Zeros()))
            model.add(layers.Activation(activations.elu))
            model.add(layers.Dropout(0.25))

        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units = 25, 
                               kernel_initializer = initializers.RandomNormal(seed = 123),
                               bias_initializer = initializers.Zeros()))
        model.add(layers.Activation(activations.elu))
        model.add(layers.Dropout(0.25))

        # Output layer
        model.add(layers.Dense(units = 1, activation = "exponential"))

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate = lr), loss = "mean_squared_error"
        )
        return model
    
    def fit(self, LR, xcol, ycol):
        """Train neural network models
        
        Parameters
        ----------
        LR : numpy.array, learning rate
        xcol: numpy.array, column names for X
        ycol: numpy.array, column names for y
        """
        models = []
        results = []
        earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
        for i in range(self.n_estimators):
            # get the training and testing sets
            X_train = self.train[i][xcol]
            X_test = self.test[i][xcol]
            y_train = self.train[i][ycol]
            y_test = self.test[i][ycol]
            
            # Standardize the features
            scalerX = MinMaxScaler().fit(X_train)
            X_train = pd.DataFrame(scalerX.transform(X_train), columns = X_train.columns.values) 
            X_test = pd.DataFrame(scalerX.transform(X_test), columns = X_train.columns.values) 

            scalery = MinMaxScaler().fit(y_train.values.reshape(-1, 1))
            y_train = scalery.transform(y_train.values.reshape(-1, 1))
            y_test = scalery.transform(y_test.values.reshape(-1, 1))

            # build the models
            ANN1 = self.build_model(LR[0])
            ANN2 = self.build_model(LR[1])
            
            # train the models
            history1 = ANN1.fit(X_train, 
                y_train, 
                epochs = 100, 
                verbose = 0,
                batch_size = 32, 
                callbacks = [earlyStop],
                validation_split = 0.2)
            history2 = ANN2.fit(X_train, 
                y_train, 
                epochs = 100, 
                verbose = 0,
                batch_size = 32, 
                callbacks = [earlyStop],
                validation_split = 0.2)

            trainr12 = np.round(r2_score(y_train, ANN1.predict(X_train)), 4)
            testr12 = np.round(r2_score(y_test, ANN1.predict(X_test)), 4)
            trainr22 = np.round(r2_score(y_train, ANN2.predict(X_train)), 4)
            testr22 = np.round(r2_score(y_test, ANN2.predict(X_test)), 4)
            
            # Save the better model
            if (trainr12 + testr12) > (trainr22 + testr22):
                models.append(ANN1)
                results.append(np.array([trainr12, testr12]))
            else:
                models.append(ANN2)
                results.append(np.array([trainr22, testr22]))
                
        self.models = models
        self.results = np.array(results)
            
        return None
    
    def predict(self, X):
        """make prediction
        
        Parameters
        ----------
        LR : pandas.DataFrame
        """
        results = []
        for i in range(self.n_estimators):
            results.append(self.models[i].predict(X))
            
        pred = np.mean(np.array(results), axis = 0)
        assert len(pred) == X.shape[0], 'wrong dim'
        
        return pred
        
    
    def get_results(self):
        """output the models results
        """
        for i in range(self.n_estimators):
            print('The training R^2 for model {} is {} and {}, respectively'.format(i, 
                                                                                    self.results[i][0], 
                                                                                    self.results[i][1]))
        return None

