/*
 *  purelin.h
 *  backprop
 *
 *  Created by George Kuan on Wed Oct 01 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#include "function.h"
#include <math.h>

class fcnPurelin : public Function
{
public:
    double evaluate(double input);
    double evalDeriv(double input);
};

