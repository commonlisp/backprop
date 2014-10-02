/*
 *  hardlim.cpp
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "hardlim.h"

/** evaluate: evaluate the hardlim at a given point
*
*/
double fcnHardlim::evaluate(double input)
{
    return (input < 0.0 ? 0.0 : 1.0);
}

/** evalDeriv: evaluate the derivative of hardlim at a given point.
* dhardlim = 0 according to Matlab
*/
double fcnHardlim::evalDeriv(double input)
{
    return (0.0);
}