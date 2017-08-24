

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cmath>
//#include <functional>
//#include "MZeta.h"
//#include "MLP.h"
#include "MMLP.h"
#include "LinearLayer.h"
#include "BALayer.h"
#include "SGD.h"
#include <random>
#include "FreeFall.h"
//using namespace std;

typedef uint64_t ulong;
using std::vector;
using std::istream;
using std::ifstream;
using std::cout;
using std::endl;


using Eigen::VectorXf;

std::ifstream dataFile("G:\\Data\\2048D4Stats.dat", std::ios::binary);
struct PosData
{
    ulong pos;
    vector<ulong> results;
};

template<typename T>
T square(const T& x)
{
    return x * x;
}

auto f = [](float x) -> float
    {
        return x < 0 ? 0 : x;
    };

    auto df = [](float x) -> float
    {
        return x < 0 ? 0 : 1;
    };

vector<PosData> data;
ulong reverseBytes(ulong i)
{

        //i = ((i & 0x5555555555555555ULL) << 1)   | ((i >> 1) & 0x5555555555555555ULL);
        //i = ((i & 0x3333333333333333ULL) << 2)   | ((i >> 2) & 0x3333333333333333ULL);
        //i = ((i & 0x0f0f0f0f0f0f0f0fULL) << 4)   | ((i >> 4) & 0x0f0f0f0f0f0f0f0fULL);
        i = ((i & 0x00ff00ff00ff00ffULL) << 8)   | ((i >> 8) & 0x00ff00ff00ff00ffULL);
        i = (i << 48) | ((i & 0xffff0000ULL) << 16) |
        ((i >> 16) & 0xffff0000ULL) | (i >> 48);
        return i;
}

template<typename T>  T&& ldim(T&& a)
{
    cout << a.rows() << " by " << a.cols() << endl;
    return a;
}

ulong readLong(istream& s)
{
    ulong x;
    s.read((char*)(ulong)&x, sizeof(x));
    x = reverseBytes(x);
    return x;
}

vector<PosData> readData(ifstream& f)
{
    vector<PosData> result;
    ulong entries = readLong(f);
    cout << entries << endl;
    result.reserve(entries);
    for(int i = 0; i < entries; i++)
    {
        ulong key = readLong(f);
        ulong num = readLong(f);
        PosData p;
        p.pos = key;
        p.results.reserve(num);
        for(int j = 0; j < num; j++)
        {
            p.results.push_back(readLong(f));
        }
        result.push_back(p);
    }
    return result;
}





struct AdamSGD
{
    MatrixXf gradEstimate, magnitudeEstimate;
    double a, b, lrate, eps;
    int iter = 1;

    AdamSGD(double a = 0.1, double b = 0.001, double learningRate = 0.001, double eps = 1e-8) : a(a), b(b), lrate(learningRate), eps(eps)
    {

    }

    void init(const MatrixXf& localGrad, double errEst)
    {
        gradEstimate = localGrad;
        magnitudeEstimate = localGrad.array().square();
    }

    template<typename Matrix, typename Acc>
    void update(Matrix& params, Acc& accum, double errEst)
    {
        MatrixXf& localGrad = accum.gradient();
        gradEstimate = (1 - a) * gradEstimate + a * localGrad;
        magnitudeEstimate = (1 - b) * magnitudeEstimate + b * localGrad.array().square().matrix();
        params.array() -= (lrate / sqrt(iter)) * gradEstimate.array() / (magnitudeEstimate.array().sqrt() + eps);
        //iter++;
    }

};


struct LabeledData
{
    MatrixXf& data;
    MatrixXf& labels;

    int size() const
    {
        return data.cols();
    }
};


void state2vec(ulong state, VectorXf& result)
{
    for(int i = 0; i < 16; i++)
    {
        result(i) = (state >> (i * 4)) & 0xF;
    }
}


int totalValue(ulong state)
{
    int result = 0;
    for(int i = 0; i < 16; i++)
    {
        int tile = (state >> (i * 4)) & 0xF;
        if(tile != 0)
            result += 1 << tile;
    }
    return result;
}
float calcScore(const PosData& pos)
{
    int total = 0;
    for(ulong u : pos.results)
        total += totalValue(u);
    float average = total * 1.0 / pos.results.size();
    return log(average);
}

auto loss = [](const auto& in, const auto& actual, const MatrixXf& target)
    {
        return sqrt((actual - target).array().square().mean());
        //return (actual - target).norm();
        //return sqrt((in.unaryExpr([](float x){return pow(2, x);}).colwise().sum().array() * (target.array() - actual.array()) / (target.array() * actual.array()) + 0.0000001).square().mean());
    };

    auto lossGrad = [](const auto& in, const auto& actual, const MatrixXf& target)
    {
        return (actual - target);
        //return (in.unaryExpr([](float x){return pow(2, x);}).colwise().sum().array().square() * (actual - target).array() / (target.array() * actual.array().cube()+ 0.0000001)).matrix();
    };

void train2048Eval()
{
    vector<PosData> data = readData(dataFile);
    cout << data.size() << endl;
    MatrixXf inputs(16, data.size()-1);
    MatrixXf outputs(1, data.size()-1);
    int index = 0;
    VectorXf temp(16);
    for(const PosData& p : data)
    {
        if(p.pos == 0) continue;
        state2vec(p.pos, temp);
        inputs.col(index) = temp;
        outputs(0, index) = calcScore(p);
        index++;
    }
    cout << index << endl;
    cout << "Loaded Training Data" << endl;
    LabeledData trainingSet = {inputs, outputs};
    typedef BALayer<decltype(f), decltype(df)> AL;
    auto id = [](float x){return x;};
    auto ONE = [](float x){return 1;};
    auto EXP = [](float x){return exp(x);};
    auto nn = MMLP< LinearLayer, AL,
                    LinearLayer, AL,
                    LinearLayer, AL,
                    LinearLayer, AL,
                    LinearLayer, BALayer<decltype(id), decltype(ONE)>>
                                (LinearLayer(64, 16), AL(16, f, df),
                                 LinearLayer(64, 64), AL(64, f, df),
                                 LinearLayer(32, 64), AL(32, f, df),
                                 LinearLayer(32, 32), AL(32, f, df),
                                 LinearLayer(1, 32),  BALayer<decltype(id), decltype(ONE)>(1, id, ONE));

    int iter = 1000000;
    driveSGD(nn, trainingSet, loss, lossGrad, [&](double err){return iter--;}, AdamSGD());
    //driveSGD(nn, trainingSet, loss, lossGrad, [&](double err){return iter--;}, FreeFall(), MomentAccum());
    std::ofstream saveLocation("G:\\Data\\5LNNWeights1MFF.txt");
    nn.save(saveLocation);
    saveLocation.flush();
    saveLocation.close();

    //out = val / net
    //out' = -val / net^2 * net'
}




using Eigen::MatrixBase;
int main()
{
    rng.seed(std::random_device()());
    train2048Eval();

/*
    typedef BALayer<decltype(f), decltype(df)> AL;
    MatrixXf data = MatrixXf::Random(10, 10000);
    auto nn = MMLP<LinearLayer, AL, LinearLayer>(LinearLayer(MatrixXf::Random(100, 10)), AL(VectorXf::Random(100), f, df), LinearLayer(MatrixXf::Random(10, 100)));
    VectorXf in = VectorXf::Random(10);
    auto learn = MMLP<LinearLayer, AL, LinearLayer>(LinearLayer(MatrixXf::Random(100, 10)), AL(VectorXf::Random(100), f, df), LinearLayer(MatrixXf::Random(10, 100)));
    MatrixXf target;
    nn.apply(data, target);
    decltype(learn)::Application a;
    decltype(learn)::Grad g;
    learn.apply(data, a);
    cout << (std::get<2>(a.outputs) - target).norm() << endl;

    LabeledData d = {data, target};

    //ADAM(learn, d, 1000);
    int iter = 150;
    //driveSGD(learn, d, loss, lossGrad, [&](double err){return iter--;}, AdamSGD());
    driveSGD(learn, d, loss, lossGrad, [&](double err){return iter--;}, FreeFall(), MomentAccum(), 256);

    learn.apply(data, a);
    cout << (std::get<2>(a.outputs) - target).norm() << endl;



    /*
    ulong entries;
    //std::cout << dataFile.
    for(int i = 0; i < 10; i++)
    {
        dataFile.read((char*)(ulong)&entries, sizeof(entries));
        entries = reverseBytes(entries);
        std::cout << entries << std::endl;
    }

    //dataFile >> entries;

    */
    return 0;
}
