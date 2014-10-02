/*
 *  parse.h
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "logsig.h"
#include "hardlim.h"
#include "tansig.h"
#include "purelin.h"

#include <iostream>
#include <fstream.h>
#include <string.h>
#include <vector.h>

using namespace std;

/** Parse the sample network input files */
class Parse {
public:
    /* Constructor reads in file and parses for outputs */
    Parse(string filename);
    ~Parse();
    
    /* Public methods */
    string getTitle();       /* Get the title in the file */
    double getLR(); /* Get the learning rate */
    double getGoal();       /* Get the desired MSE threshold */
    int getMaxEpochs();     /* Get the maximum epochs before give up */

    vector<Function*> getFunctions(); /* Get activation functions */
    vector<vector<double> > getInputVectors();
                                      /* Get input vectors */
    vector<vector<double> > getTargets();      /* Get the targets of each sample */
    vector<int> getNumNeuronsInLayer(); /* Get number of neurons in each layer */
    
private:
    /* Disable copy constructor and assignment */
    Parse(const Parse&);
    Parse operator=(const Parse&);

    /* Private methods */
    int parsefile(ifstream& input);
    int parsefunctions(string bufstr);
    int parsesamples(string bufstr);
    
    /* Private fields */

    const int MAXLINELEN;

    string title;
    double learningRate;
    double goal;
    double multiplier;
    int momentum;
    int epochs;
    int interval;

    vector<Function*> functions;
    vector<vector<double> > inputvectors;
    vector<vector<double> > targetMatrix;
    vector<int> neuronsInLayer;
};
    

