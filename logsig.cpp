/*
 *  logsig.cpp
 *  backprop
 *
 *  Created by George Kuan on Sat Sep 27 2003.
 *  Copyright (c) 2003 George Kuan. All rights reserved.
 *
 */

#include "logsig.h"

/** evaluate: evaluate the logsig at a given point
 *
 */
double fcnLogsig::evaluate(double input)
{
    return (1.0 / (1.0 + exp(-input)));
}

/** evalDeriv: evaluate the derivative of logsig at a given point.
 *
 */
double fcnLogsig::evalDeriv(double input)
{
    return (1.0 / (4.0 * pow(cosh(input / 2.0), 2.0)));
}