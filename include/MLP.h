#ifndef MLP_H
#define MLP_H
#include "DenseLayer.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <vector>


using std::vector;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
using Eigen::VectorXf;



template <class MatT>
Eigen::Matrix<typename MatT::Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime>
pseudoinverse(const MatT &mat, typename MatT::Scalar tolerance = typename MatT::Scalar{1e-4}) // choose appropriately
{
    typedef typename MatT::Scalar Scalar;
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    Eigen::Matrix<Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime> singularValuesInv(mat.cols(), mat.rows());
    singularValuesInv.setZero();
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tolerance)
        {
            singularValuesInv(i, i) = Scalar{1} / singularValues(i);
        }
        else
        {
            singularValuesInv(i, i) = Scalar{0};
        }
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}

MatrixXf checkMult(const MatrixXf& A, const MatrixXf& B)
{
    printf("%lld by %lld * %lld by %lld\n", A.rows(), A.cols(), B.rows(), B.cols());
    return A * B;
}

//template<typename F, dF>
class MLP
{

    vector<DenseLayer> layers;
public:


    void randInit(vector<int> widths)
    {
        for(int i = 1; i < widths.size(); i++)
        {
            layers.emplace_back(widths[i - 1], widths[i]);
        }
        for(DenseLayer& L : layers)
        {
            L.weights = MatrixXf::Random(L.weights.rows(), L.weights.cols());
            L.bias = VectorXf::Random(L.bias.rows()) * 2;
            //cout << L.weights << endl;
        }
    }
    struct GradAccum
    {
        vector<MatrixXf> wGrads;
        vector<VectorXf> bGrads;
    };

    void apply(const VectorXf& in, vector<VectorXf>& layerOutputs)
    {
        layers[0].apply(in, layerOutputs[0]);
        for(int i = 1; i < layers.size(); i++)
        {
            layers[i].apply(layerOutputs[i - 1], layerOutputs[i]);
        }
    }


/*
    void backprop(const VectorXf& in, const VectorXf& out, GradAccum& result)
    {
        vector<VectorXf> layerOutputs(layers.size());
        vector<VectorXf> layerInternals(layers.size());

        layers[0].apply(in, layerOutputs[0], layerInternals[0])
        for(int i = 1; i < layers.size(); i++)
        {
            layers[i].apply(layerOutputs[i - 1], layerOutputs[i], layerInternals[i]);
        }
        vector<VectorXf> outDerivs(layers.size());
        outDerivs[layers.size() - 1] = layerOutputs[layers.size() - 1] - out;
        for(int i = layers.size() - 1; i >= 1; i--)
        {
            outDerivs[i - 1] = layers[i].transpose() * (outDerivs[i].array() * layerInternals[i].unaryExpr(dF));
        }


    }
*/

    void layerSolve(DenseLayer& layer, const MatrixXf& A, const MatrixXf& target)
    {

    }
    template<typename F, typename IF>
    void linProp(const MatrixXf& in, const MatrixXf& desiredOut, F f, IF inv)
    {
        vector<MatrixXf> layerOutputs(layers.size());
        vector<MatrixXf> layerLinOutputs(layers.size());
        vector<MatrixXf> layerTargets(layers.size());
        auto propagate = [&](int layer, const MatrixXf& data)
        {
            int rows = layers[layer].bias.rows() + 1;
            layerOutputs[layer].resize(rows, data.cols());
            auto workingPart = layerOutputs[layer].topRows(rows - 1);
            //cout << "Hi" << endl;
            layers[layer].applyLin(data.topRows(data.rows() - 1), layerLinOutputs[layer]);
            f(layerLinOutputs[layer], workingPart);
            layerOutputs[layer].bottomRows(1).setConstant(1);
        };
        auto layerSolve = [&](int i, const MatrixXf& A, const MatrixXf& target)
        {
            //cout << __LINE__ << endl;
            //A.bottomRows(1).setConstant(1);
            //cout << A.rows() << " " << A.cols() << endl;
            MatrixXf normEqM = A * A.transpose();
            //cout << "Formed normal equation" << endl;
            MatrixXf solution = normEqM.lu().solve(A * target.transpose()).transpose();
            //cout << "Solved" << endl;
            layers[i].weights = solution.leftCols(solution.cols() - 1);
            layers[i].bias = solution.col(solution.cols() - 1);
        };
        auto targetSolve = [&](int i, const MatrixXf& currentOut, const MatrixXf& nextTarget)
        {
            /*
            auto solver = layers[i + 1].weights.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            decltype(solver)::SingularValuesType singInv = solver.singularValues();

            double thresh = 1e-3;
            for(long i = 0; i < singInv.cols(); i++)
            {
                if(singInv(i) > thresh)
                   singInv(i) = 1.0 / singInv(i);
                else singInv(i) = 0;
             }
             MatrixXf x(layers[i + 1].weights.cols(), layers[i + 1].weights.rows());
             x.setZero();
             x.diagonal() = singInv;
             */
            MatrixXf psuedoInverse = pseudoinverse<MatrixXf>(layers[i + 1].weights, 1e-3);//solver.matrixV()* x * solver.matrixU().adjoint();
            MatrixXf tout;
            layers[i + 1].applyLin(layerOutputs[i].topRows(layerOutputs[i].rows() - 1), tout);
            //cout << nextTarget.rows() << " " << nextTarget.cols()
                //<< " " << currentOut.rows() << " " << currentOut.cols() << endl;
                            //target = psuedoInverse * nextTarget
                            //layer * (currentOut + deltaX) + bias = nextTarget
                            //layer * deltaX = nextTarget - layer * currentOut - bias
                            //deltaX = psuedoInverse * (nextTarget - layer * currentOut - bias);
                            //cout << "Should be 0:" << (((layers[i + 1].weights * layerOutputs[i].topRows(layerOutputs[i].rows() - 1)).colwise() + layers[i + 1].bias) - nextTarget).norm() << endl;
                            MatrixXf deltaX = psuedoInverse *  ((nextTarget - tout));
            layerTargets[i] = layerOutputs[i].topRows(layerOutputs[i].rows() - 1) + deltaX;

            //layerTargets[i] = psuedoInverse * (nextTarget.colwise() - layers[i + 1].bias);
            //MatrixXf& A = layers[i + 1].weights;
            //cout << (A * psuedoInverse * A - A).norm() << endl;
            //cout << (psuedoInverse * A * psuedoInverse - psuedoInverse).norm() << endl;
           // MatrixXf test;
           // layers[i + 1].applyLin(layerTargets[i], test);
            //cout << (test - nextTarget).norm() << endl;
            //layers[i + 1].applyLin(layerOutputs[i].topRows(layerOutputs[i].rows() - 1), test);
            //cout << (test - currentOut).norm() << endl;
            //cout << (currentOut - nextTarget).norm() << endl;

            inv(layerTargets[i]);
        };
        propagate(0, in);
        for(int i = 1; i < layers.size(); i++)
            propagate(i, layerOutputs[i - 1]);
        int last = layers.size() - 1;
        //cout << __LINE__ << endl;
        layerTargets[last] = desiredOut;
        //cout << __LINE__ << endl;
        inv(layerTargets[last]);
        //cout << __LINE__ << endl;
        //cout << layerTargets[last].leftCols(5) << endl;
        layerSolve(last, layerOutputs[last - 1], layerTargets[last]);
        //cout << __LINE__ << endl;
        for(int i = last - 1; i >= 1; i--)
        {
            targetSolve(i, layerLinOutputs[i + 1],layerTargets[i + 1]);
            //cout << layerTargets[i].leftCols(5) << endl;
            layerSolve(i, layerOutputs[i - 1], layerTargets[i]);
        }
        targetSolve(0, layerLinOutputs[1], layerTargets[1]);
       //layerTargets[0].leftCols(5) << endl;
        layerSolve(0, in, layerTargets[0]);
    }

        template<typename A, typename F>
        void apply(const A& in, A& out, F f)
        {
            A lastOut = in;
            A result;

            for(int i = 0; i < layers.size(); i++)
            {
                //std::cout << layers[i].weights << std::endl;
                layers[i].applyLin(lastOut, result);
                f(result, lastOut);
            }
            out = lastOut;
        }






};



#endif // MLP_H
