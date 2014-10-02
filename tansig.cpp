/*
 *  tansig.cpp
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "tansig.h"

/** evaluate: evaluate the tansig at a given point
*
*/
double fcnTansig::evaluate(double input)
{
    return (tanh(input));
}

/** evalDeriv: evaluate the derivative of tansig at a given point.
* 
*/
double fcnTansig::evalDeriv(double input)
{
    return (1 - pow(tanh(input),2));
}