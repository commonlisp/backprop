/*
 *  layer.cpp
 *  
 *
 *  Created by George Kuan on Tue Sep 23 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */

#include "layer.h"

Layer::Layer(int _numNeurons, int _numInputs, double _learningRate, Function* _activationFunction) 
    : numNeurons(_numNeurons), numInputs(_numInputs), learningRate(_learningRate),
      activationFunction(_activationFunction)
{
    /** Create weight matrix according to the number of neurons specified */
    /**
     * The number of "destination neurons" determine the number of rows
     * whereas the number of sources determine the number of columns.
     * Hagan 2-10
     */
    /* randomize weights matrix at this point? */
    randomizeWeights();
}

/** Layer destructor cleans up any allocation messes we've created manually.
 * The destructor needs to delete at least the weightMatrix that was allocated
 * by the constructor.
 */
Layer::~Layer()
{

}

/** randomizeWeights() : Pseudorandomly generate weights and initialize
 * weightMatrix and biases vector.
 */
int Layer::randomizeWeights()
{
    vector<double> row;
    
    for (int i = 0; i < numNeurons; i++) {
        for (int j = 0; j < numInputs; j++) {
            double randnum = (double)(rand()/((double)(10.0*(RAND_MAX + 1))));
            // cout << "Random num: " << randnum << endl;
            row.push_back(randnum);
        }
        weightMatrix.push_back(row);
        row.clear(); // Clear the temporary row holder for generation of next row

        double randbias = (double)(rand()/((double)(10.0*(RAND_MAX + 1))));
        biases.push_back(randbias);
    }

    return 0;
}

/** setLearningRate(...): Set a new learning rate for this layer
 */
int Layer::setLearningRate(double newLearningRate)
{
    learningRate = newLearningRate;   
    return 0;
}

int Layer::setMomentum(double newMomentum)
{
    return 0;
}

/** forwardPropagate(...): Propagate an input forward through the
 * layer using weightMatrix.
 */ 
vector<double> Layer::forwardPropagate(vector<double> inputVec)
{
    /** Forward propagation of input through network
    *  a^{0} = p
    *  a^{m+1} = f^{m+1}(W^{m+1}a^{m} + b^{m+1}) for m = 0, 1, ..., M - 1
    * a = a^{M}
    */

    int inputLen = inputVec.size();
    currentInput = inputVec; 

    double result = 0.0;

#ifdef DEBUG
    cout << "input vector: ";
    for (int i = 0; i < inputVec.size(); i++) {
        cout << inputVec[i] << " ";
    }
    cout << endl;
#endif

    nValues.clear();
    outputValues.clear(); // Erase any junk because we're propagating something new
    for (int i = 0; i < numNeurons; i++) {
        
        result = 0.0;
        
        for (int j = 0; j < inputLen; j++) {
            result += inputVec[j]*weightMatrix[i][j];
        }
        result += biases[i]; //weightMatrix[i][inputLen - 1];
                             // Add in the bias, each neuron has a separate bias
        nValues.push_back(result);
        //cout << "Result: " << result << " ";
        result = activationFunction->evaluate(result);
        //cout << "Evaluated result: " << result << endl;
        outputValues.push_back(result);
    }

#ifdef DEBUG
    cout << "layer output vector: [ ";
    for (int i =0 ; i < numNeurons; i++) {
        cout << outputValues[i] << " ";
    }
    cout << " ] " << endl;
#endif
    return outputValues;  // Return the output values but there is still an
                          // internal copy so we won't have to recompute
}

/** backPropoagate: propagate sensitivities backward through the network.
 *
 */
vector<double> Layer::backPropagate(vector<double> nextLayerWgtSenProd)
{
    int outputlen = outputValues.size();
    
    if (outputlen == 0) {
        cerr << "Layer::backPropagate() : no sensitivities to backpropagation." << endl;
        abort();
    } else if (outputlen != numNeurons) {
        cerr << "Layer::backPropagate() : output vector length does not correspond"
             << " to number of neurons in the layer." << endl;
        abort();
    }

    sensitivities.clear(); /* We are calculating fresh sensitivities so
                              hopefully there
                              is no stale data. */

    vector<double> row;
    vector<double> weightMatrixSensitivityProd;
                             /* The matrix-vector product of the present layer's
                                weight matrix and current layer's sensitivity is
                                what the previous layer really needs for
                                backpropagation. */
    int numInputs = weightMatrix.size();
    double dotprod = 0.0;    /* An element of the product of Jacobian for this
                                layer and the weight matrix and sensitivity
                                vector for the next layer over */
    
    
    /** Open code for sensitivity propagation
     * The sensitivity vector s is a # of neurons x 1 vector
     * the weightMatrix W is a # of neurons x # inputs matrix
     * The Jacobian F is a #input x #inputs square matrix
     * The sensitivity of this layer is s = F*W^T*s^(next layer)
     * Consequently, the dimensions of the resulting sensitivity
     * vector is also #neurons x 1.
     * To compute the sensitivity vector of the current layer,
     * iterate through the #neurons and then the #inputs
     * taking the dot products of the next layer sensitivity
     * vector and the matrix product of the Jacobian and
     * transposed weightMatrix. 
     */
    for (int i = 0; i < numInputs; i++) {
        dotprod = 0.0;
        //cout << "Propagating sensitivity" << endl;
        for (int j = 0; j < numNeurons; j++) {
            dotprod += (activationFunction->evalDeriv(nValues[i]) *
                                nextLayerWgtSenProd[j]);
            //cout << dotprod << " ";
            
        }
        //cout << endl; 
        sensitivities.push_back(dotprod);
    }

    
    
    /* dotprod is now an element in the product of this layer's weight matrix
        and this layer's sensitivities */
    for (int i = 0; i < numInputs; i++) {
        dotprod = 0.0;
        for (int j = 0; j <  numNeurons; j++) {
            dotprod += weightMatrix[j][i] * sensitivities[j];
        }
        weightMatrixSensitivityProd.push_back(dotprod);
    }

    updateWeights();    


    
    return weightMatrixSensitivityProd;
}

/** backPropagateError(...): Last layer propagates error via s = -2F(n)(t-a)
 * During the operation of the neural net, this function should only be called
 * once per backpropagation. The bulk of the backpropagation should be handled
 * by the main backPropagation function that deals with all the layers except
 * the last layer. This backPropagateError function handles the last layer
 * where there are no sensitivities in a following layer to propagate. 
 */
vector<double> Layer::backPropagateError(vector<double> targets)
{
    int targetlen = targets.size();

    if (targetlen != numNeurons) {
        cerr << "Layer::backPropagateError() : target vector length and number of neurons mismatch"
             << endl;
        abort();
    }
    
    sensitivities.clear();

    cout << "backPropagateError.nValues: [ ";
    for (int i = 0; i < nValues.size(); i++) {
        cout << nValues[i] << " ";
    }
    cout << " ] " << endl;
    
    cout << "backPropagateError.outputValues: [ ";
    for (int i = 0; i < outputValues.size(); i++) {
        cout << outputValues[i] << " ";
    }
    cout << " ] " << endl;

    cout << "backPropagateError.targets: [ ";
    for (int i = 0; i < targets.size(); i++) {
        cout << targets[i] << " ";
    }
    cout << " ] " << endl;    
//    cout << "end sensitivities" << endl;
    for (int i = 0; i < numNeurons; i++) {
        double newsensitivity =
          -2.0 * (activationFunction->evalDeriv(nValues[i]) *
                                        (targets[i] - outputValues[i]));
        // cout << "new sensitivity " << i << " " << newsensitivity << " ";
        sensitivities.push_back(newsensitivity);
    }

    

    vector<double> weightMatrixSensitivityProd;
    double dotprod = 0.0;
    
    /* dotprod is now an element in the product of this layer's weight matrix
        and this layer's sensitivities */
    for (int i = 0; i < numInputs; i++) {
        dotprod = 0.0;
        for (int j = 0; j <  numNeurons; j++) {
            dotprod += weightMatrix[j][i] * sensitivities[j];
        }
        weightMatrixSensitivityProd.push_back(dotprod);
    }

    cout << "Product: [ ";
    for (int i = 0; i < weightMatrixSensitivityProd.size(); i++) {
        cout << weightMatrixSensitivityProd[i] << " ";
    }
    cout << " ] " << endl;
    
    updateWeights();
    
    return weightMatrixSensitivityProd;
}

/** updateWeights(...): Helper function for backPropagation. Update the weights
 *  using approximate gradient descent.
 */
int Layer::updateWeights()
{
    //cout << "Old and new weights " << endl;
    for (int i = 0; i < numNeurons; i++) {
        for (int j = 0; j < numInputs; j++) {
            /* Use gradient descent to compute new weightMatrix entry */
            //cout << weightMatrix[i][j] << " ";
            weightMatrix[i][j] =
               weightMatrix[i][j] - learningRate *
                                      sensitivities[i] * currentInput[j];

            //cout << weightMatrix[i][j] << " ";
        }
        /* Use gradient descent to compute new bias */
        //cout << endl << "Biases: " << biases[i] << " ";
        biases[i] = biases[i] - learningRate * sensitivities[i];
        //cout << endl << "New Bias: " << biases[i] << " ";
    }
    //cout << endl;
    return 0;
}
            