/*
 *  layer.h
 *  
 *
 *  Created by George Kuan on Tue Sep 23 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */

#include "Function.h"
#include <vector.h>

/** Layer: a layer of neurons in the neural network
 * with methods for forward propagation of input,
 * backpropagation of sensitivities, and resettable
 * learning rate. 
 */
class Layer {

    public:
        /* Constructors and Destructors */
        Layer(int _numNeurons, int _numInputs, double _learningRate, Function* _activationFunction);
        ~Layer();

        /* Public Methods */

        /* Reset learning rate */
        int setLearningRate(double newLearningRate);
        
        int setMomentum(double newMomentum);
        
        /* forward propagate the input data */
        vector<double> forwardPropagate(vector<double> inputVec);

        /* backpropagate the error for the output layer. This function
         * must be called from the network because a layer itself
         * does not know whether it is the output or hidden layer.
         */
        vector<double> backPropagateError(vector<double> targets);

        /* backpropagation for hidden layers: It takes the next layer's
           weight matrix and sensitivity vector product as an argument */
        vector<double> backPropagate(vector<double> nextLayerWgtSensProd);
        
    private:
        /* Disable assignment and copy constructor */
        Layer operator= (const Layer&);
        Layer(const Layer&);

        /* Update weights after backpropagation: helper to backPropagate */
        int updateWeights();
        /* Randomize weight matrix and biases only upon construction */
        int randomizeWeights();

        /* Member fields */
        int numNeurons; /* Number of neurons in the layer, determined by the
                           constructor. */
        int numInputs; /* Number of input also determined during construction */
        double learningRate; /* Rate of approximate gradient descent for
                                weight updates */
        
        Function* activationFunction; /* Object corresponding to activation function
                                         for the layer */
        vector<double> activationValues; /* Threshold value for activation */
        vector<double> outputValues; /* Store for output values so that they won't
                                        have to be recomputed for backpropagation */
        vector<double> sensitivities; /* Sensitivities for backpropagation */
        vector<vector<double> > weightMatrix; /* Weight vectors for each neuron in
                                                 the layer */
        vector<double> biases; /* Biases for each neuron in the layer */
        vector<double> currentInput; /* current input vector for use with gradient
                                        descent */
        vector<double> nValues;

};