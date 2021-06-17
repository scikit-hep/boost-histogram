#include <boost/histogram.hpp>
#include <iostream>

int main() {
    namespace bh = boost::histogram;

    auto h = bh::make_histogram(bh::axis::category<int>{});

    std::cout << bh::algorithm::sum(h) << std::endl;
}
