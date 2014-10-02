/*
 *  main.cpp
 *  backprop
 *
 *  Created by George Kuan on Fri Sep 26 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */

#include <iostream>
#include <stdlib.h>
#include "network.h"
#include "logsig.h"
#include "parse.h"

void print_usage(char* progname);

int main(int argc, char* argv[])
{
    /** Test out Layer class
     *
     */
    vector<double> dummyInput(5, 99);
    vector<double> test_targets(6, 1);
    /*
    fcnLogsig* dummyFunction = new fcnLogsig;
    Layer dummyLayer(3, dummyInput.size(), 0.5, dummyFunction);
    */
    vector<int> neuronsInLayer(4,6); // 6 neurons in 4 layers
    neuronsInLayer[1] = 4;
    vector<Function*> dummyFunctions;

    /** Four layers of neurons and therefore four
     * activation functions.
     */
    for (int i = 0; i < 4; i++) {
        dummyFunctions.push_back(new fcnLogsig);
    }

    Network dummyNet(dummyInput.size(), neuronsInLayer,
                  dummyFunctions);
    for (int i = 0 ; i < 15; i++) {
        dummyNet.trainingStep(dummyInput, test_targets);
        cout << "MSE: " << dummyNet.oldgetMSE()
            << endl;
    }
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Open and print out parsed parameters for verification */
    Parse infile(argv[1]);
    cout << "Title: " << infile.getTitle() << endl;
    cout << "Learning Rate: " << infile.getLR() << endl;
    cout << "Goal: " << infile.getGoal() << endl;
    cout << "Epochs: " << infile.getMaxEpochs() << endl;

    vector<vector<double> > inputVectors = infile.getInputVectors();
    cout << "Input vectors: ";
    for (unsigned int i = 0; i < inputVectors.size(); i++) {
        for (unsigned int j = 0; j < inputVectors[i].size(); j++) {
            cout << inputVectors[i][j] << " ";
        }
        cout << endl;
    }
    
    vector<vector<double> > targetMat = infile.getTargets();
    
    cout << "Target vector: ";
    for (unsigned int j = 0; j < targetMat.size(); j++) {
        for (unsigned int i = 0; i < targetMat[j].size(); i++) {
            cout << targetMat[j][i] << " ";
        }
    }
    cout << endl;

    Network myNet(inputVectors[0].size(), infile.getNumNeuronsInLayer(),
                  infile.getFunctions());
    double mse = infile.getGoal() + 1.0;
    int epoch = 0;

    
    while (mse > infile.getGoal() || epoch == infile.getMaxEpochs()) {
        myNet.trainingCycle(inputVectors, targetMat);
        //mse = myNet.getMSE(targetVals);
        cout << "Epoch: " << epoch << endl;
        epoch++;
    }
    return 0;
}

void print_usage(char* progname)
{
    cout << "Usage: " << progname << " [filename of input]" << endl;
}

