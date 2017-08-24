#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;
//template<typename F, typename dF>
class DenseLayer
{
    public:
    MatrixXf weights;
    VectorXf bias;
    //F func;
    //dF funcDeriv;

    public:
    //DenseLayer(int size, F&& f, dF&& df) : func(f), funcDeriv(df)
    DenseLayer(int inSize, int outSize)
    {
        weights.resize(outSize, inSize);
        bias.resize(outSize);
    }
    template<typename A, typename B>
    void applyLin(const A& in, B& out)
    {
        //printf("%lld by %lld * %lld by %lld\n", weights.rows(), weights.cols(), in.rows(), in.cols());
        out.noalias() = weights * in;
        //printf("%lld vs %lld\n", out.rows(), bias.rows());
        out.colwise() += bias;
    }

    void apply(const VectorXf& in, VectorXf& out)
    {
        out = (weights * in + bias).array().max(0);
    }

    void apply(const VectorXf& in, VectorXf& out, VectorXf& internal)
    {
        out = (internal = weights * in + bias).array().max(0);
    }


/*
    void backprop(const VectorXf& in, const VectorXf& internValues, const VectorXf& outDerivs, VectorXf& inDerivs, MatrixXf& wGradSum, VectorXf& bGradSum)
    {
        VectorXf internGrads = (outDerivs.array() * (internValues.array() >= 0);
        inDerivs.noalias() = weights.transpose() * internGrads.matrix();
        wGradSum.noalias() += internGrads.matrix() * in.transpose();
        bGradSum.noalias() += internGrads;


    }
*/

};

#endif // DENSELAYER_H
