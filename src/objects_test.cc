#include "distributions.hpp"
#include "gtest/gtest.h"

#include "exp/user/stb/slam/objects.hh"

namespace {
using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

class ObjectsTest : public ::testing::Test {
protected:
  ObjectsTest()
      : epsilon_(1e-10),
        colorMeanPrior_(new GaussianModelGivenSigma(nullptr, 0, 1)),
        colorPrecisionPrior_(new ChisqModel(1, 1)),
        background_(new Background2D(colorMeanPrior_, colorPrecisionPrior_)),
        squarePrior_(new SquarePrior(1.0, new UniformModel(1, 25),
                                     new UniformModel(0, Constants::pi / 2),
                                     colorMeanPrior_, colorPrecisionPrior_)),
        circlePrior_(new CirclePrior(1.0, new UniformModel(1, 25),
                                     colorMeanPrior_, colorPrecisionPrior_)) {
    BOOM::GlobalRng::rng.seed(8675309);
  }
  double epsilon_;
  Ptr<GaussianModelGivenSigma> colorMeanPrior_;
  Ptr<ChisqModel> colorPrecisionPrior_;
  std::shared_ptr<Background2D> background_;
  std::shared_ptr<SquarePrior> squarePrior_;
  std::shared_ptr<CirclePrior> circlePrior_;
};

TEST_F(ObjectsTest, Point) {
  Point2D point(3, 7);
  EXPECT_DOUBLE_EQ(point.x(), 3);
  EXPECT_DOUBLE_EQ(point.y(), 7);
}

TEST_F(ObjectsTest, Rotation) {
  // Rotate 90 degrees around the origin.
  Rotation rot(BOOM::Constants::pi / 2.0, Point2D(0, 0));

  Point2D start(3, 0);
  Point2D end = rot(start);
  EXPECT_NEAR(end.x(), 0.0, epsilon_);
  EXPECT_NEAR(end.y(), 3.0, epsilon_);

  // Rotate start 45 degrees around (2, 1).
  Rotation rot45(BOOM::Constants::pi / 4.0, Point2D(2, 1));
  end = rot45(start);
  // After centering, the point is at 1, -1.  Rotating a further 45 degrees
  // puts it at (sqrt(2), 0) in centered coordinates, so 2 + sqrt(2), 1.0 in
  // original coordinates.
  EXPECT_NEAR(end.x(), 2.0 + sqrt(2), epsilon_);
  EXPECT_NEAR(end.y(), 1.0, epsilon_);
}

TEST_F(ObjectsTest, Background2D) {
  int color = background_->simulateColor(GlobalRng::rng);
  // plogis(1.1) * 256 = 192.0666
  // plogis(1.1 - 3 * .25) * 256  = 150.1741
  EXPECT_NEAR(color, 192, 40);
}

TEST_F(ObjectsTest, Square) {
  Square aligned(13, 4, 2, 0, squarePrior_);
  EXPECT_TRUE(aligned.contains(Point2D(12, 3)));
  EXPECT_TRUE(aligned.contains(Point2D(14, 3)));
  EXPECT_TRUE(aligned.contains(Point2D(12, 5)));
  EXPECT_TRUE(aligned.contains(Point2D(14, 5)));

  EXPECT_FALSE(aligned.contains(Point2D(11.9, 3)));
  EXPECT_FALSE(aligned.contains(Point2D(14.1, 3)));
  EXPECT_FALSE(aligned.contains(Point2D(11.9, 5)));
  EXPECT_FALSE(aligned.contains(Point2D(14.1, 5)));

  EXPECT_FALSE(aligned.contains(Point2D(12, 2.9)));
  EXPECT_FALSE(aligned.contains(Point2D(14, 2.9)));
  EXPECT_FALSE(aligned.contains(Point2D(12, 5.1)));
  EXPECT_FALSE(aligned.contains(Point2D(14, 5.1)));

  double angle = BOOM::Constants::pi / 4;
  Square rotated(1, 1, 2, angle, squarePrior_);
  EXPECT_TRUE(rotated.contains(Point2D(1, 1 + sqrt(2))));
  EXPECT_TRUE(rotated.contains(Point2D(1, 1 - sqrt(2) + epsilon_)));
  EXPECT_TRUE(rotated.contains(Point2D(1 + sqrt(2), 1)));
  EXPECT_TRUE(rotated.contains(Point2D(1 - sqrt(2) + epsilon_, 1)));

  BoundingBox box = rotated.boundingBox();
  EXPECT_TRUE(box.contains(Point2D(1, 1 + sqrt(2))));
  EXPECT_TRUE(box.contains(Point2D(1, 1 - sqrt(2) + epsilon_)));
  EXPECT_TRUE(box.contains(Point2D(1 + sqrt(2), 1)));
  EXPECT_TRUE(box.contains(Point2D(1 - sqrt(2) + epsilon_, 1)));

  rotated.setAngle(.25);
  EXPECT_EQ(rotated.angle(), .25);
  rotated.setAngle((Constants::pi / 2) + .27);
  EXPECT_EQ(rotated.angle(), .27);
  rotated.setAngle(-.25);
  EXPECT_EQ(rotated.angle(), (Constants::pi / 2) - .25);
}

TEST_F(ObjectsTest, Circle) {
  Circle circle(3, 7, 2, circlePrior_);
  EXPECT_TRUE(circle.contains(Point2D(3, 7)));
  EXPECT_TRUE(circle.contains(Point2D(5, 7)));
  EXPECT_TRUE(circle.contains(Point2D(1, 7)));
  EXPECT_TRUE(circle.contains(Point2D(3, 9)));
  EXPECT_TRUE(circle.contains(Point2D(3, 5)));
}

} // namespace
