#include "distributions.hpp"
#include "gtest/gtest.h"

#include "exp/user/stb/slam/objects.hh"

namespace {
using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

class BoxUnionTest : public ::testing::Test {
protected:
  BoxUnionTest() {
    BOOM::GlobalRng::rng.seed(8675309);
    epsilon = 1e-10;
  }
  double epsilon;
};

TEST_F(BoxUnionTest, AddBox) {
  BoundingBox ten(0, 10, 0, 10);
  BoundingBox five(5, 10, 5, 10);
  BoxUnion bu(ten, five);
}

} // namespace
