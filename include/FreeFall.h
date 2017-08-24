#ifndef FREEFALL_H
#define FREEFALL_H
#include <Eigen/Dense>
#include <cmath>
//#include <unsupported/Eigen/SpecialFunctions>
using Eigen::MatrixXf;

struct MomentAccum
{
    MatrixXf sum;
    MatrixXf squareSum;
    int count = 0;

    template<typename M>
    void init(const M& localGrad)
    {
        sum = localGrad;
        squareSum = localGrad.array().square();
    }

    template<typename M>
    void accumulate(const M& sampleGrads)
    {
        sum += sampleGrads;
        squareSum += sampleGrads.array().square().matrix();
        count++;
    }

    void reset()
    {
        sum.setZero();
        squareSum.setZero();
    }

    MatrixXf& gradient()
    {
        return sum;
    }

    auto variance() const
    {
        return ((squareSum.array() - sum.array().square() / count) / (count - 1));
    }

};

struct FreeFall
{
    MatrixXf gradEstimate;
    MatrixXf gradVar;
    MatrixXf stepSizes;
    MatrixXf uncertaintyGrowths;
    MatrixXf deltaGrad, k;
    float uFalloff = 0.01;
    float minStepMult = 0.5;
    float maxStepMult = 2;
    float eps = 1e-8;

    void init(const MomentAccum& dat, double errEst)
    {
        gradEstimate = dat.sum / dat.count;
        gradVar = dat.variance() / dat.count;
        stepSizes = MatrixXf::Constant(gradEstimate.rows(), gradEstimate.cols(), 0.001);
        uncertaintyGrowths = gradVar;
        uncertaintyGrowths.setZero();

    }

    template<typename Matrix>
    void update(Matrix& params, MomentAccum& m, double errEst)
    {

        //Diagonal Covariance Kalman filter


        k = gradVar.array() / (gradVar.array() +  m.variance() / m.count);
        auto badFloat = [](float x)
        {
            return !std::isfinite(x);
        };
        //if(k.unaryExpr(badFloat).any())
            //cout << "knan" << endl;
        //cout << "kcalc" << endl;
        deltaGrad = (m.sum / m.count - gradEstimate).array() * k.array();
        //cout << "deltagrad" << endl;
        /*
        stepSizes.array() *= (deltaGrad.array() / gradEstimate.array()).unaryExpr([this](float x) -> float
                                                                                  {
                                                                                      if(std::isinf(x)) return 1;
                                                                                      if(x > 1) return maxStepMult;
                                                                                      if(x < -1) return minStepMult;
                                                                                        x = (x + 1) / 2;
                                                                                       return x*x*(3 - 2*x) * 1.1;
                                                                                  });
                                                                                  */


        //cout << "stepSizes" << endl;
        gradEstimate += deltaGrad;
        gradVar.array() *= (1 - k.array());
        //cout << "gradVar" << endl;

        //params.array() -= stepSizes.array() * gradEstimate.array().unaryExpr([](float x){return (x > 0) - (x < 0);});

        //cout << "update" << endl;

        //increase uncertainty after move
        gradVar += uncertaintyGrowths * stepSizes.norm();
        //cout << "gradVar2" << endl;

        //TODO FIX! need to fit uncertaintyGrowths, probably max-likelihood-ish
        uncertaintyGrowths.array() = (1 - uFalloff) * uncertaintyGrowths.array() + uFalloff * deltaGrad.array().square() / (stepSizes.array().square().sum() + eps);
        cout << gradEstimate(0, 0) << " " << sqrt(gradVar(0, 0)) << " ";
            //if(uncertaintyGrowths.unaryExpr(badFloat).any())
               // cout << "ugnan" << endl;
        //cout << "UGupdate" << endl;


    }





};

#endif // FREEFALL_H
