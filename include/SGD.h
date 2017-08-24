#ifndef SGD_H
#define SGD_H
#include <array>
#include <Eigen/Dense>
#include <iostream>
#include "MMLP.h"
typedef uint64_t ulong;
using std::vector;
using std::istream;
using std::ifstream;
using std::cout;
using std::endl;
using std::get;


using Eigen::VectorXf;
using Eigen::MatrixXf;

std::mt19937 rng;
template<typename Data>
void selectBatch(const Data& d, MatrixXf& inResult, MatrixXf& outResult, int batchSize = 256)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, d.size() - 1);
    for(int i = 0; i < batchSize; i++)
    {
        int selection = dist(rng);
        inResult.col(i) = d.data.col(selection);
        outResult.col(i) = d.labels.col(selection);
    }

}

template<typename Net, typename Algo, typename Acc>
struct DriveHelper
{
    Net& net;
    std::array<Acc, Net::size()>& accums;
    double errorEst;
    std::array<Algo, Net::size()>& algoStates;

    template<int N>
    void operator() ()
    {
        algoStates[N].update(get<N>(net.layers).params, accums[N], errorEst);
    }
};

template<typename Net, typename Algo, typename Acc>
struct SGDInit
{
    std::array<Algo, Net::size()>& algoStates;
    std::array<Acc, Net::size()>& accums;
    double errorEst;
    template<int N>
    void operator() ()
    {
        algoStates[N].init(accums[N], errorEst);
    }
};
struct DefaultAccum
{
    MatrixXf grad;

    template<typename M>
    void accumulate(const M& sampleGrads)
    {
        grad += sampleGrads;
    }

    operator MatrixXf& ()
    {
        return grad;
    }

    MatrixXf& gradient()
    {
        return grad;
    }

    void reset()
    {
        grad.setZero();
    }

    template<typename Matrix>
    void init(Matrix& m)
    {
        grad = m;
    }
};

template<typename Accums, typename Grad>
struct Init2
{
    Accums& accs;
    Grad& g;
    template<int N>
    void operator() ()
    {
        accs[N].init(get<N>(g.grads));
    }
};
template<typename Net, typename Data, typename LossF, typename DLossF, typename Algo, typename Acc = DefaultAccum, typename F>
void driveSGD(Net& n, const Data& data, LossF error, DLossF dError, F stopping, Algo algo, Acc accum = DefaultAccum(), int batchSize = 256)
{
    MatrixXf in(data.data.rows(), batchSize);
    MatrixXf targetOut(data.labels.rows(), batchSize);
    typename Net::Application app;
    typename Net::Grad localGrad;
    std::array<Acc, n.size()> accums;
    accums.fill(accum);
    typename Net::PropMem mem;
    MatrixXf outgrad;
    double errorEst;
    selectBatch(data, in, targetOut, batchSize);
    n.apply(in, app);
    outgrad = dError(in, app.output(), targetOut);
    n.backprop(in, app, localGrad, outgrad, mem);
    Init2<decltype(accums), decltype(localGrad)> in2 = {accums, localGrad};
    static_for<0, n.size()>() (in2);
    auto backpropBatch = [&]()
    {
        selectBatch(data, in, targetOut, batchSize);
        n.apply(in, app);
        for(Acc& acc: accums) acc.reset();
        outgrad = dError(in, app.output(), targetOut);
        //n.backprop(in, app, localGrad, outgrad, mem);
        n.backprop(in, app, accums, outgrad, mem);
        //cout << accums[1].grad(0, 0) << endl;
        errorEst = error(in, app.output(), targetOut);
    };
    std::array<Algo, n.size()> algStates;
    algStates.fill(algo);
    backpropBatch();
    SGDInit<Net, Algo, Acc> init = {algStates, accums, errorEst};
    static_for<0, n.size()>() (init);
    int count = 0;
    do
    {
        count++;
        backpropBatch();
        DriveHelper<Net, Algo, Acc> dh = {n, accums, errorEst, algStates};
        static_for<0, n.size()>()(dh);

        //if(count % 1000 == 0)
            cout << errorEst << endl;
        //cout << accums[1].grad(0, 0) << endl;
    }
    while(stopping(errorEst));
}

#endif // SGD_H
