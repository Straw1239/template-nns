#ifndef MMLP_H
#define MMLP_H
#include <tuple>
#include <vector>
#include <iostream>

using std::vector;
using std::tuple;
using std::get;
using std::endl;
using std::cout;
using std::ostream;
template <class T1, class ...T>
struct first
{
    typedef T1 type;
};

template <class T1, class ...T>
struct last
{
    typedef typename last<T...>::type type;
};

template <class T1>
struct last<T1>
{
    typedef T1 type;
};

template <int First, int Last>
struct static_for
{
    template <typename Fn>
    void operator()(Fn& fn) const
    {
        if (First < Last)
        {
            fn.operator()<First>();
            static_for<First+1, Last>()(fn);
        }
    }
};

template <int N>
struct static_for<N, N>
{
    template <typename Fn>
    void operator()(Fn const& fn) const
    { }
};



template<typename... Layers>
class MMLP
{
    #define layer(i) get<i>(layers)


public:
    typedef typename first<Layers...>::type::In In;
    typedef typename last<Layers...>::type::Out Out;
    typedef tuple<typename Layers::In...> PropMem;
    tuple<Layers...> layers;
    struct Application
    {
        tuple<typename Layers::Out...> outputs;

        Out& output()
        {
            return get<sizeof...(Layers) - 1>(outputs);
        }
    };

    struct Grad
    {


        tuple<typename Layers::Grad...> grads;




    };
private:



     struct Applier
        {
            Application& result;
            tuple<Layers...>& layers;
            template<int index> void operator() () const
            {
                layer(index).apply(get<index - 1>(result.outputs), get<index>(result.outputs));
            }

        };

    struct Backpropper
    {
        const Application& a;
        Grad& result;
        tuple<typename Layers::In...>& mem;
        tuple<Layers...>& layers;

        template<int index> void operator() () const
        {
            constexpr int last = size() - 1;
            constexpr int i = last - index;
            layer(i).backprop(get<i - 1>(a.outputs), get<i + 1>(mem), get<i>(result.grads), get<i>(mem));
        }
    };

public:

    static constexpr int size()
    {
        return sizeof...(Layers);
    }

    struct Saver
    {
        tuple<Layers...>& layers;
        ostream& out;
        template<int N> void operator() ()
        {
            get<N>(layers).save(out);
        }
    };

    void save(ostream& location)
    {
        Saver s = {layers, location};
        static_for<0, size()>() (s);
    }



    MMLP(const Layers&&... layers) : layers(layers...)
    {

    }





    void apply(const In& in, Out& out)
    {
        Application a;
        apply(in, a);
        out = get<size() - 1>(a.outputs);
    }


    void apply(const In& in, Application& result)
    {
        layer(0).apply(in, get<0>(result.outputs));
        Applier x = {result, layers};
        static_for<1, size()>()(x);
        //applyP(result, std::make_integer_sequence<int, size() - 1>());
    }

    void backprop(const In& in, const Application& a, Grad& result, const Out& outgrads, tuple<typename Layers::In...> mem = tuple<typename Layers::In...>())
    {
        constexpr int last = size() - 1;

        layer(last).backprop(get<last - 1>(a.outputs), outgrads, get<last>(result.grads), get<last>(mem));
        //cout << "bp1" << endl;
        Backpropper x = {a, result, mem, layers};
        static_for<1, last>()(x);
        //cout << "bp2" << endl;
        //backPropP(a, result, mem, std::make_integer_sequence<int, size() - 2>());
        layer(0).backprop(in, get<1>(mem),get<0>(result.grads), get<0>(mem));


    }

    template<typename Acc>
    struct FBackpropper
    {
        const Application& a;
        std::array<Acc, size()>& accums;
        tuple<typename Layers::In...>& mem;
        tuple<Layers...>& layers;

        template<int index> void operator() () const
        {
            constexpr int last = size() - 1;
            constexpr int i = last - index;
            layer(i).backprop(get<i - 1>(a.outputs), get<i + 1>(mem), accums[i], get<i>(mem));
        }
    };

    template<typename Acc>
    void backprop(const In& in, const Application& a, std::array<Acc, size()>& accums, const Out& outgrads, tuple<typename Layers::In...> mem = tuple<typename Layers::In...>())
    {
        constexpr int last = size() - 1;

        layer(last).backprop(get<last - 1>(a.outputs), outgrads, accums[last], get<last>(mem));
        FBackpropper<Acc> x = {a, accums, mem, layers};
        static_for<1, last>()(x);
        layer(0).backprop(in, get<1>(mem), accums[0], get<0>(mem));
    }

    struct Incrementer
    {
        tuple<Layers...>& layers;
        const Grad& g;

        template<int n> void operator() ()
        {
            layer(n).adjust(get<n>(g.grads));
        }
    };

    void adjust(const Grad& g)
    {
        Incrementer i = {layers, g};
        static_for<0, size()>()(i);
    }





    #undef layer
};

#endif // MMLP_H
