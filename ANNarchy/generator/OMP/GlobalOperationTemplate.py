max_template = """
// Computes the maximum value of an array
double max_value(const std::vector<double> &array)
{
    double max = array[0];
    for(int i=0; i<array.size(); i++)
    {
        if(array[i] > max)
            max = array[i];
    }

    return max;
}
"""
min_template = """
// Computes the minimum value of an array
double min_value(const std::vector<double> &array)
{
    double min = array[0];
    for(int i=0; i<array.size(); i++)
    {
        if(array[i] < min)
            min = array[i];
    }

    return min;
}
"""
mean_template = """
// Computes the mean value of an array
double mean_value(const std::vector<double> &array)
{
    double sum = 0.0;
    #pragma omp parallel reduction(+:sum)
    for(int i=0; i<array.size(); i++)
    {
        sum += array[i];
    }

    return sum/(double)(array.size());
}
"""
norm1_template = """
// Computes the L1-norm of an array
double norm1_value(const std::vector<double> &array)
{
    double sum = 0.0;
    #pragma omp parallel reduction(+:sum)
    for(int i=0; i<array.size(); i++)
    {
        sum += fabs(array[i]);
    }

    return sum;
}
"""
norm2_template = """
// Computes the L2-norm (Euclidian) of an array
double norm2_value(const std::vector<double> &array)
{
    double sum = 0.0;
    #pragma omp parallel reduction(+:sum)
    for(int i=0; i<array.size(); i++)
    {
        sum += pow(array[i], 2.0);
    }

    return sqrt(sum);
}
"""