#include "distributions.hpp"
#include "gtest/gtest.h"

#include "exp/user/stb/slam/screen2d.hh"
#include "exp/user/stb/slam/simulation2d.hh"

namespace {
using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

class Screen2DTest : public ::testing::Test {
protected:
  Screen2DTest()
      : epsilon_(1e-10),
        color_mean_prior_(new GaussianModelGivenSigma(nullptr, 0, 1)),
        color_precision_prior_(new ChisqModel(1, 1)),
        background_(
            new Background2D(color_mean_prior_, color_precision_prior_)),
        circlePrior_(new CirclePrior(1.0, new UniformModel(1, 25),
                                     color_mean_prior_,
                                     color_precision_prior_)),
        squarePrior_(new SquarePrior(1.0, new UniformModel(1, 25),
                                     new UniformModel(0, Constants::pi / 2),
                                     color_mean_prior_,
                                     color_precision_prior_)) {
    BOOM::GlobalRng::rng.seed(8675309);
  }

  double epsilon_;
  Ptr<GaussianModelGivenSigma> color_mean_prior_;
  Ptr<ChisqModel> color_precision_prior_;
  std::shared_ptr<Background2D> background_;
  std::shared_ptr<CirclePrior> circlePrior_;
  std::shared_ptr<SquarePrior> squarePrior_;
};

TEST_F(Screen2DTest, Screen) {
  SimulationCourse sim(1000, 50, circlePrior_, squarePrior_, background_);
  sim.simulate();

  EXPECT_EQ(sim.flightLengthPixels(), 1000);
  EXPECT_EQ(sim.screenWidthPixels(), 50);

  Screen2D screen(40, &sim);

  EXPECT_FALSE(screen.done());
  EXPECT_EQ(screen.bottom(), 0);
  EXPECT_EQ(screen.top(), 39);

  EXPECT_EQ(0, screen.left());
  EXPECT_EQ(50 - 1, screen.right());

  screen.advance(100);
  EXPECT_EQ(screen.bottom(), 100);
  EXPECT_EQ(screen.top(), 139);
  EXPECT_EQ(0, screen.left());
  EXPECT_EQ(50 - 1, screen.right());

  EXPECT_EQ(screen.nextFilename(), "screen_image_00000000.png");
  EXPECT_EQ(screen.nextFilename(), "screen_image_00000001.png");
  EXPECT_EQ(screen.nextFilename(), "screen_image_00000002.png");

  double grayValue = screen(2, 107, Screen2D::Scale::GRAY);
  double logitValue = screen(2, 107, Screen2D::Scale::LOGIT);
  EXPECT_DOUBLE_EQ(BOOM::qlogis((grayValue + 1.0) / 257), logitValue)
      << "gray: " << grayValue << " logit: " << logitValue;
}

} // namespace
