/*
 *  logsig.h
 *  backprop
 *
 *  Created by George Kuan on Sat Sep 27 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */
#ifndef LOGSIG_H
#define LOGSIG_H 1

#include "function.h"
#include <math.h>

/** Implementation of the transfer function logsig
 * Algorithm: logsig(x) = 1/(1+exp(-x))
 */
class fcnLogsig : public Function
{
public:
    double evaluate(double input);
    double evalDeriv(double input);
};

#endif

