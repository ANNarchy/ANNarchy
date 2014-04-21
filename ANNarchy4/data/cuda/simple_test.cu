#include "simple_test.h"
#include <stdio.h>

__global__ void helloCuda()
{
	printf("Hello from your device :-)");
}