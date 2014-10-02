/*
 *  network.cpp
 *  
 *
 *  Created by George Kuan on Tue Sep 23 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "network.h"

/** Network class default constructor
 *
 */
Network::Network(int _numInputs, vector<int> numNeurons,
                 vector<Function*> activationFcnType) :
      numInputs(_numInputs), defLearningRate(0.4)
{
    unsigned int numLayers = numNeurons.size();
          /* vector::size() returns unsigned int */

    /** It is important to check whether the number of layers
     * implied by the numNeurons vector corresponds with that
     * implied by the activationFcnType vector. Otherwise,
     * the user is likely to be making an error.
     */
    if (numLayers == activationFcnType.size()) {
        /** Create the layers corresponding to the parameters */
        for (unsigned int i = 0; i < numLayers; i++) {
            networkLayers.push_back(new Layer(numNeurons[i],
                                              numInputs, defLearningRate,
                                              activationFcnType[i]));
        }
    } else {
        cerr << "**Network constructor: numLayers "
             << numLayers << " and activationFcnType vector size "
             << activationFcnType.size() << " differ."
             << endl;
        abort();
    }

    srand(time(NULL)); // Seed random just once for whole network for randomizing
                       // the weight matrices
}

/** Network destructor
 * Clean up all the layers
 */
Network::~Network()
{
    unsigned int numLayers = networkLayers.size();
    for (unsigned int i = 0; i <  numLayers; i++) {
        delete (networkLayers[i]);
    }
}

/** setLearningRate(...): Reset the learning rates for
 * all the layers to a given learning rate.
 */
int Network::setLearningRate(double newLearningRate)
{
    int numLayers = networkLayers.size();

    for (int i = 0; i < numLayers; i++) {
        networkLayers[i]->setLearningRate(newLearningRate);
    }

    return 0;
}

/** trainingStep(...): A public interface to train the
 * network on a single input vector and vector of targets.
 * Also reports the MSE after completion of training.
 */ 
int Network::trainingStep(vector<double> _inputVector,
                          vector<double> _target)
{
      inputVector = _inputVector;
      targets.clear();
      targets = _target;

      if (_inputVector.size() < 1) {
          cerr << "trainingStep(): inputVector size "
               << _inputVector.size() << " malformed."
               << endl;
          return 1;
      } 

      /* cout << "Current input: ";
      for (unsigned int i = 0;
           i < _inputVector.size(); i++) {
          cout << inputVector[i] << " ";
      }
      cout << endl; */
      
      forwardPropagate();
      backPropagate();

      /* cout << "Current outputs: ";
      for (unsigned int i = 0;
           i < currentOutputs.size(); i++) {
          cout << currentOutputs[i] << " ";
      }
      cout << endl;*/
      
      return 0;
}

/** trainingCycle(...): A public interface to the network that
 * trains the network on a set of input vectors and target vectors
 */
int Network::trainingCycle(vector<vector<double> > inputVectors,
                           vector<vector<double> > targetMat)
{
    vector<double> outputs;
    
    if (inputVectors.size() < 1) {
        cerr << "trainingCycle: Invalid inputvector. Size "
             << inputVectors.size() << endl;
        return 1;
    }

    for (unsigned int i = 0; i < inputVectors.size(); i++) {
        trainingStep(inputVectors[i], targetMat[i]);
        //cout << "currentOutputs size " << currentOutputs.size() << endl;
        outputs.push_back(currentOutputs[0]);
    }

    vector<double> targetVec;

    //cout << "Targetmatrix: "; 
    for (unsigned int i = 0; i < targetMat.size(); i++) {
        for (unsigned int j = 0; j < targetMat[i].size(); j++) {
            //cout << targetMat[i][j] << " ";
            targetVec.push_back(targetMat[i][j]);
        }
    }

    double mse = getMSE(targetVec, outputs);
    cout << "MSE: " << mse << endl;
    
    return 0;
}

/** Iterate through the layers propagating the current data.
 *
 */
vector<double> Network::forwardPropagate()
{
    int numLayers = networkLayers.size();


    
    vector<double> outputVals(inputVector);

/*    cout << "input: [ ";
    for (int i = 0; i < outputVals.size(); i++) {
        cout << outputVals[i] << " ";
    }
    cout << " ] " << endl;*/
    
    for (int i = 0; i < numLayers; i++) {
        outputVals = networkLayers[i]->forwardPropagate(outputVals);
    }

/*    cout << "outputVals: [ ";
    for (int i = 0; i < outputVals.size(); i++) {
        cout << outputVals[i] << " ";
    }
    cout << " ] " << endl;*/
    
    currentOutputs = outputVals;
    
    return outputVals;
}

/** Network::backPropagate() propagate sensitivities back through layers
 *
 */
vector<double> Network::backPropagate()
{
    unsigned int numLayers = networkLayers.size();
    unsigned int numOutputs = currentOutputs.size();

    vector<double> sensitivities;
    vector<double> error;
    
    if (numOutputs != targets.size()) {
        cerr << "Network:backPropagate() : length of target vector "
             << targets.size() 
             << " not equal to length of output vector "
             << numOutputs << endl;
        abort();
    }

    error.clear();
    sensitivities.clear();

    /** Obtain error in current propagation t - a */
    for (unsigned int i = 0; i < numOutputs; i++) {
        error.push_back(targets[i] - currentOutputs[i]);
    }

    /** Backpropagate error */
    
    /** Backpropagate error on last layer */
    sensitivities = networkLayers[numLayers - 1]->backPropagateError(targets);
    
    /** Backpropagate sensitivities on prior layers */
    for (int i = numLayers - 2; i >= 0; i--) {
        sensitivities = networkLayers[i]->backPropagate(sensitivities);
    }

    return sensitivities;
}

/** oldgetMSE(): Compute and return the mean square error for this layer
*/
double Network::oldgetMSE()
{
    double mse = 0.0; /* cumulative mean squared error */
    unsigned int numOutputs = currentOutputs.size();

    if (numOutputs != targets.size()) {
        cerr << "Targets and output vectors length mismatch: targets "
        << targets.size() << " and outputs " << numOutputs << endl;
        abort();
    }

    for (unsigned int i = 0; i < numOutputs; i++) {
        mse += (targets[i] - currentOutputs[i]) *
        (targets[i] - currentOutputs[i]);
    }

    return mse;
}

/** getMSE(): Compute and return the mean square error for this layer
 */
double Network::getMSE(vector<double> targetsVector,
                       vector<double> outputs)
{
    double mse = 0.0; /* cumulative mean squared error */
    unsigned int numOutputs = outputs.size();
    
    if (numOutputs != targetsVector.size()) {
        cerr << "Targets and output vectors length mismatch: targets "
             << targetsVector.size() << " and outputs " << numOutputs << endl;
        abort();
    }
    
    for (unsigned int i = 0; i < numOutputs; i++) {
        
        //cout << outputs[i] << " ";
        mse += (targetsVector[i] - outputs[i]) *
               (targetsVector[i] - outputs[i]);
    }

    /*cout << "targetsVector: " << "[ ";
    for (unsigned int i = 0; i < numOutputs; i++) {
        cout << targetsVector[i] << " ";
    }
    cout << "] " << endl;

    cout << "outputs: [ ";
    for (unsigned int i = 0; i < numOutputs; i++) {
        cout << outputs[i] << " ";
    }
    cout << "] " << endl;  */
    
    return mse;
}

/** getErrorVector(): Returns the target - output error vector
 */
vector<double> Network::getErrorVector()
{
    vector<double> error;
    int numOutputs = currentOutputs.size();

    for (int i = 0; i < numOutputs; i++) {
        error.push_back(targets[i] - currentOutputs[i]);
    }

    return error;
}
