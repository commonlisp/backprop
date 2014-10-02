/*
 *  parse.cpp
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */

#include "parse.h"

using namespace std;

/* Default constructor */
Parse::Parse(string filename)
: MAXLINELEN(256)
{
    ifstream input(filename.c_str(), ios::in);

    if (input.bad()) {
        cerr << "Error opening " << filename << ". " << endl;
        exit(1);
    }

    parsefile(input);
    input.close();
    
}

/* Destructor */
Parse::~Parse()
{
    unsigned functionsLen = functions.size();

    for (unsigned int i = 0; i < functionsLen; i++) {
        delete functions[i];
    }
}

/** Parse out elements from input file useful to neural net simulation */
int Parse::parsefile(ifstream& input)
{
    char buf[MAXLINELEN];
    
    while (!input.eof()) {
        input.getline(buf, 256);
        string bufstr(buf);

        /* Parse title */
        if (bufstr.find("title",0) != string::npos) {
            int titlestartindex = bufstr.find("(", 0) + 1;
            int titleendindex = bufstr.find(")", titlestartindex);
            title = bufstr.substr(titlestartindex,
                                  titleendindex - titlestartindex);
        } else {
            int start = bufstr.find("=", 0) + 2;
            // assuming trailing space
            int end = bufstr.length();
            // parameter runs to end of line

            /* Parse learning rate */
            if (bufstr.substr(0,2) == "LR") {
                learningRate = atof(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("momentum",0) != string::npos) {
                momentum = atoi(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("goal",0) != string::npos) {
                goal = atof(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("epochs",0) != string::npos) {
                epochs = atoi(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("interval",0) != string::npos) {
                interval = atoi(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("multiplier",0) != string::npos) {
                multiplier = atof(bufstr.substr(start,end).c_str());
            } else if (bufstr.find("layers",0) != string::npos) {
                parsefunctions(bufstr);
            } else if (bufstr.find("samples",0) != string::npos) {
                cout << "Processing samples." << endl;
                while (!input.eof()) {
                    input.getline(buf, MAXLINELEN);
                    bufstr = buf;
                    parsesamples(bufstr);
                }
            } else {
                cerr << "Unknown line encountered " << bufstr << endl;
            }
        }
    }
    return 0; 
}

/** Parse functions from input line */
int Parse::parsefunctions(string bufstr)
{
    unsigned int startfcn = bufstr.find("(", 0);
    while (startfcn != string::npos) {

        if (bufstr.find("logsig", startfcn) != string::npos) {
            functions.push_back(new fcnLogsig);
        } else if (bufstr.find("hardlim", startfcn) !=
                   string::npos) {
            functions.push_back(new fcnHardlim);
        } else if (bufstr.find("tansig", startfcn) !=
                   string::npos) {
            functions.push_back(new fcnTansig);
        } else if (bufstr.find("purelin", startfcn) !=
                   string::npos) {
            functions.push_back(new fcnPurelin);
        } else {
            cerr << "Skipping function. " << endl;
        }

        /* Process the number of inputs for that layer here */
        
        
        int endfcn = bufstr.find(")", startfcn);
        int startNum = 0;
        
        for (startNum = endfcn - 1; isdigit(bufstr[startNum]); startNum--);
        neuronsInLayer.push_back(atoi(bufstr.substr(startNum + 1,
                                                    endfcn - startNum).c_str()));
        cout << "Neurons in Layer " << atoi(bufstr.substr(startNum + 1,
                                                          endfcn - startNum).c_str())
            << endl;
        
        startfcn = bufstr.find("(", endfcn);
    }
    return 0;
}

/** Parse sample */
int Parse::parsesamples(string bufstr)
{
    // cout << "Look at line " << bufstr << endl;
    unsigned int pairparen = bufstr.find("(",0);
    if (pairparen != string::npos) {
        //cout << "Found opening (" << endl;
        int inputparen = bufstr.find("(",pairparen + 1);
        int inputcloseparen = bufstr.find(")", inputparen);

        if (inputparen != string::npos && inputcloseparen != string::npos) {
        vector<double> samp;
        int startdigit = inputparen + 1;
        int digitlen = 0;

        for (int i = startdigit; i < inputcloseparen; i++) {
            cout << "Reading " << bufstr[i] << endl;
            if (isdigit(bufstr[i]) || bufstr[i] == '.') {
                digitlen++;
            } else if (bufstr[i] == ' ') {
                samp.push_back(
                               atof(bufstr.substr(startdigit,
                                                  digitlen).c_str()));
                startdigit = i + 1;
            } else {
                cerr << "Reading samples. Unexpected input character "
                << bufstr[i] << endl;
            }

        }
        samp.push_back(atof(bufstr.substr(startdigit, digitlen).c_str()));
        inputvectors.push_back(samp);
        

        /* Read in targets */
        unsigned int targetparen =
            bufstr.find("(", startdigit);
        unsigned int targetcloseparen =
            bufstr.find(")", targetparen);
        if (targetparen != string::npos && targetcloseparen != string::npos) {
            double targetvalue =
                atof(bufstr.substr(
                                   targetparen + 1,
                                   targetcloseparen - targetparen + 1).c_str());

            vector<double> targets;
            targets.push_back(targetvalue);
            targetMatrix.push_back(targets);
        }
        }
    }
    return 0;
}

string Parse::getTitle()
{
    return title;
}

double Parse::getLR()
{
    return learningRate;
}

double Parse::getGoal()
{
    return goal;
}

int Parse::getMaxEpochs()
{
    return epochs;
}

vector<Function*> Parse::getFunctions()
{
    return functions;
}

vector<vector<double> > Parse::getInputVectors()
{
    return inputvectors;
}

vector<vector<double> > Parse::getTargets()
{
    return targetMatrix;
}

vector<int> Parse::getNumNeuronsInLayer()
{
    return neuronsInLayer;
}
