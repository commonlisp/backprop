/*
 *  hardlim.h
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "function.h"

class fcnHardlim : public Function
{
public:
    double evaluate(double input);
    double evalDeriv(double input);
};
