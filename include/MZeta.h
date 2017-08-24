#ifndef MZETA_H_INCLUDED
#define MZETA_H_INCLUDED
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>

template<typename M> M zeta(M x)
{

}

template<typename M> M gamma(M x)
{

}
template<typename M, typename F> M trapIntegrate(F f, long double a, long double b, int segments)
{
    M result = M::Zero();
    result += 0.5 * (f(a) + f(b));
    long double h = (b - a) / segments;
    for(int i = 1; i < segments; i++)
    {
        result += f(i*(b - a) / segments + a);
    }
    result *= h;
    return result;
}

template<typename M, typename F> M integrate(F f, long double a, long double b, int seg, int depth)
{

    if(depth == 0) return trapIntegrate<M, F>(f, a, b, 1 << seg);
    int scale = 1 << (2 * depth);
    M result = scale * integrate<M, F>(f, a, b, seg, depth - 1) - integrate<M, F>(f, a, b, seg - 1, depth - 1);
    result /= scale - 1;
    return result;
}




#endif // MZETA_H_INCLUDED
