#include "utils.h"
#include <random>

#ifdef _DEBUG
double *buf_max;
#endif

double randm()
{
    static   std::random_device generator;
    //static   std::default_random_engine generator;
    static   std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
};

