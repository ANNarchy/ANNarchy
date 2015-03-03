global_operation_templates = {
    'max' : """
// Computes the maximum value of an array
double max_value(const double* array, int n)
{
    double max = array[0];
    for(int i=0; i<n; i++)
    {
        if(array[i] > max)
            max = array[i];
    }

    return max;
}
    """,
    'min' : """
// Computes the minimum value of an array
double min_value(const double* array, int n)
{
    double min = array[0];
    for(int i=0; i<n; i++)
    {
        if(array[i] < min)
            min = array[i];
    }

    return min;
}
    """,
    'mean' : """
// Computes the mean value of an array
double mean_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += array[i];
    }
    return sum/(double)n;
}
    """,
    'norm1' : """
// Computes the L1-norm of an array
double norm1_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += fabs(array[i]);
    }

    return sum;
}
    """,
    'norm2' : """
// Computes the L2-norm (Euclidian) of an array
double norm2_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += pow(array[i], 2.0);
    }

    return sqrt(sum);
}
    """
}