#include "build/ANNarchy.h"

int main(int argc, char* argv[])
{
    ANNarchy* sim = new ANNarchy();

    sim->run(1);

    delete sim;
    return 0;
}
