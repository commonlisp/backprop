/*
 *  function.h
 *  backprop
 *
 *  Created by George Kuan on Fri Sep 26 2003.
 *  Copyright (c) 2003 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FUNCTION_H
#define FUNCTION_H 1

/** Function: Abstract class for all types of activation functions */
class Function 
{
    public:
        Function();
        
        virtual double evaluate(double input) = 0;
        virtual double evalDeriv(double input) = 0;

    private:
        /* Disable assignment and copy constructor */
        Function operator=(const Function&);
        Function(const Function&);
        
};

#endif

