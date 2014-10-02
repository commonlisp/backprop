/*
 *  network.h
 *  
 *
 *  Created by George Kuan on Tue Sep 23 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include <vector.h>
#include "layer.h"

/** Neural network base class
 * relies on Layer for neuron layer
 * functionality. Offers single input
 * and target vector training as well
 * as multiple input and target vectors
 * training.
 */
class Network {

public:
    /* Constructor and destructor */
    Network(int _numInputs, vector<int> numNeurons, 
            vector<Function*> activationFcnType);
    ~Network();

    /* Public methods */

    /* Reset learning rate for entire network */
    int setLearningRate(double newLearningRate);

    /* Single input and target vector training */
    int trainingStep(vector<double> _inputVector, vector<double> _target);

    /* Multiple input and target training */
    int trainingCycle(vector<vector<double> > inputVectors,
                      vector<vector<double> > targetsVector);
    
    // int batchTrainingCycle();
    double oldgetMSE();
    double getMSE(vector<double> targetsVector,
                  vector<double> outputs);
    vector<double> getErrorVector();

private:
    /* Disable assignment and copy constructor */
    Network operator=(const Network&);
    Network(const Network&);

    /* Private helper methods */
    vector<double> forwardPropagate(); /* Forward propagate inputs through layers */
    vector<double> backPropagate();    /* Backpropagate sensitivities through layers */
    
    int randomizeWeights();            /* Randomize weight matrix and biases during
                                          construction */
    vector<vector<double> > getWeights(); /* Obtain current weight matrix */
    int setWeights(vector<vector<double> > newWeights);
                                          /* Manually set current weight matrix */

    /* Private member fields */
    int numInputs;                /* Length of input vector for network */
    const double defLearningRate; /* Default learning rate for
                                     gradient descent */
    
    vector<Layer*> networkLayers; /* Layers in network */
    vector<double> inputVector; /* Input vector currently being processed */
    vector<double> currentOutputs; /* Last computed outputs */
    vector<double> targets; /* Target values */
    

};


