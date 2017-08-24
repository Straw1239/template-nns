#ifndef BALAYER_H
#define BALAYER_H
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;

template<typename F, typename dF>
class BALayer
{
    F f;
    dF df;

public:
    VectorXf params;
    typedef MatrixXf In;
    typedef MatrixXf Out;
    typedef VectorXf Grad;

    BALayer(const VectorXf& bias, F f, dF df) :  f(f), df(df), params(bias)
    {

    }

    BALayer(int size, F f, dF df) : f(f), df(df), params(VectorXf::Random(size))
    {

    }

    void apply(const MatrixXf& in, MatrixXf& out)
    {
        out = (in.colwise() + params).unaryExpr(f);
    }

    void backprop(const In& in, const Out& outGrads, Grad& grad, In& inGrads)
    {
        //cout << in.rows() << " " << in.cols() << endl;
        //cout << outGrads.rows() << " " << outGrads.cols() << endl;
        inGrads = (in.colwise() + params).unaryExpr(df).array() * outGrads.array();
        grad = inGrads.rowwise().sum();
        //cout << "babp" << endl;
    }

    template<typename Acc>
    void backprop(const In& in, const Out& outGrads, Acc& accum, In& inGrads)
    {

        inGrads = (in.colwise() + params).unaryExpr(df).array() * outGrads.array();
        for(int i = 0; i < inGrads.cols(); i++)
        {
            accum.accumulate(inGrads.col(i));
            //accum.accumulate(inGrads.rowwise().sum()); //inGrads is also the bias gradient
        }

    }

    void save(ostream& out)
    {
        out << params.rows() << endl;
        out << params << endl;
    }

    void adjust(const Grad& g)
    {
        params -= g;
    }

};

#endif // BALAYER_H
