#ifndef LINEARLAYER_H
#define LINEARLAYER_H
#include <Eigen/Dense>
using Eigen::MatrixXf;

class LinearLayer
{

public:
    MatrixXf params;
    typedef MatrixXf In;
    typedef MatrixXf Out;
    typedef MatrixXf Grad;

    LinearLayer(const MatrixXf& weights) : params(weights)
    {

    }

    LinearLayer(int a, int b) : params(MatrixXf::Random(a, b))
    {

    }

    void apply(const MatrixXf& in, MatrixXf& out)
    {
        out = params * in;
    }

    void backprop(const MatrixXf& in, const MatrixXf& outGrads, MatrixXf& grad, MatrixXf& inGrads)
    {
        inGrads = params.transpose() * outGrads;
        //cout << "llingrad" << endl;
        //cout << outGrads.rows() << " " << outGrads.cols() << endl;
        //cout << in.rows() << " " << in.cols() << endl;
        grad = outGrads * in.transpose();
        //cout << grad.rows() << endl;
        //cout << params.rows() << endl;

        //cout <<"llbp" << endl;
    }

    template<typename Acc>
    void backprop(const MatrixXf& in, const MatrixXf& outGrads, Acc& accum, MatrixXf& inGrads)
    {
        inGrads = params.transpose() * outGrads;
        for(int i = 0; i < outGrads.cols(); i++)
        {
            //MatrixXf g = outGrads.col(i) * in.transpose().row(i);
            //cout << g.rows() << " "<< g.cols() << endl;
            //cout << accum.grad.rows() << " " << accum.grad.cols() << endl;
            accum.accumulate(outGrads.col(i) * in.transpose().row(i));
            //cout << "hi" << endl;
        }
        //accum.accumulate(outGrads * in.transpose());
    }

    void save(ostream& out)
    {
        out << params.rows() << " " << params.cols() << endl;
        out << params << endl;
    }
    void adjust(const Grad& g)
    {

       params -= g;
    }


};

#endif // LINEARLAYER_H
