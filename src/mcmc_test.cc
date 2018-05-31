#include "distributions.hpp"
#include "gtest/gtest.h"

#include "exp/user/stb/slam/object_sampler.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/simulation2d.hh"

#include "Models/ChisqModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/UniformModel.hpp"

#include "LinAlg/Matrix.hpp"
#include "cpputil/seq.hpp"
#include "stats/moments.hpp"

namespace {
using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

class McmcTest : public ::testing::Test {
protected:
  McmcTest()
      : screenPixelWidth_(200), screenPixelHeight_(200),
        flightPixelLength_(200),
        background_(
            new Background2D(new GaussianModelGivenSigma(nullptr, 0, .1),
                             new ChisqModel(20, 1.0))),
        circlePrior_(
            new CirclePrior(2.0 / (screenPixelHeight_ * screenPixelWidth_),
                            new UniformModel(1, 25),
                            new GaussianModelGivenSigma(nullptr, 0, .001),
                            new ChisqModel(20, .01))),
        squarePrior_(new SquarePrior(
            2.0 / (screenPixelHeight_ * screenPixelWidth_),
            new UniformModel(1, 25), new UniformModel(0, Constants::pi / 2.0),
            new GaussianModelGivenSigma(nullptr, 0, .001),
            new ChisqModel(20, .01))),
        sim_(flightPixelLength_, screenPixelWidth_, circlePrior_, squarePrior_,
             background_),
        screen_(screenPixelHeight_, &sim_), sampler_(background_, 10),
        epsilon_(1e-8) {
    BOOM::GlobalRng::rng.seed(8675309);
    sampler_.addObjectType(circlePrior_);
    sampler_.addObjectType(squarePrior_);
    background_->setColorMean(2);
    background_->setColorSd(1);
  }

  // Width of the screen, in pixels.  This number of pixels will be also be seen
  // on the height of the screen in a moving window.
  int screenPixelWidth_;
  int screenPixelHeight_;
  int flightPixelLength_;
  std::shared_ptr<Background2D> background_;
  std::shared_ptr<CirclePrior> circlePrior_;
  std::shared_ptr<SquarePrior> squarePrior_;
  SimulationCourse sim_;
  Screen2D screen_;
  ObjectSampler sampler_;

  double epsilon_;
};

//---------------------------------------------------------------------------
TEST_F(McmcTest, SquareTranslationLogLikelihood) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior_));

  sim_.addSquare(square);
  sim_.simulate();

  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  // Verify that the correct location has the highest log likelihood.

  // Make a region that contains the square and a few pixels on either side.
  BoundingBox likelihood_region(60, 120, 100, 160);
  Vector logLikelihoods;
  // Evaluate log likelihood at the right answer.
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, likelihood_region, objects, background_));

  square->moveTo(Point2D(79, 120));
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, likelihood_region, objects, background_));
  square->moveTo(Point2D(81, 120));
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, likelihood_region, objects, background_));
  square->moveTo(Point2D(80, 119));
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, likelihood_region, objects, background_));
  square->moveTo(Point2D(80, 121));
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, likelihood_region, objects, background_));

  EXPECT_TRUE(logLikelihoods[0] == max(logLikelihoods))
      << "log likelihood values: " << logLikelihoods - max(logLikelihoods);
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, MoveSquare) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();

  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  // Check that the square stays at the right answer.
  int niter = 200;
  for (int i = 0; i < niter; ++i) {
    square->proposeMove(GlobalRng::rng, screen_, BoundingBox(screen_), objects,
                        background_);
  }
  EXPECT_LT(distance(square->center(), Point2D(80, 120)), 1.0)
      << "Center should be close to (80, 120), actual value: "
      << square->center() << endl
      << square->logMessages();

  // Now check that the square finds the right answer from elsewhere.
  square->moveTo(Point2D(40, 40));
  square->proposeGridSearch(GlobalRng::rng, screen_, BoundingBox(screen_),
                            objects, background_);
  for (int i = 0; i < 5 * niter; ++i) {
    square->proposeMove(GlobalRng::rng, screen_, BoundingBox(screen_), objects,
                        background_);
  }
  EXPECT_LT(distance(square->center(), Point2D(80, 120)), 1.0)
      << "Center should be close to (80, 120), actual value: "
      << square->center() << endl
      << square->logMessages();
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, RotateSquare) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 23, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  square->setAngle(0);
  for (int i = 0; i < 100; ++i) {
    square->proposeRotate(BOOM::GlobalRng::rng, screen_, square->boundingBox(),
                          objects, background_);
  }

  EXPECT_NEAR(square->angle(), Constants::pi / 8, .05) << square->logMessages();
}
//---------------------------------------------------------------------------
TEST_F(McmcTest, RotateSquareWithSlice) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 23, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  int niter = 100;

  square->setAngle(0);
  Vector angleDraws(niter);
  for (int i = 0; i < niter; ++i) {
    square->sliceRotate(BOOM::GlobalRng::rng, screen_, square->boundingBox(),
                        objects, background_);
    angleDraws[i] = square->angle();
  }

  EXPECT_NEAR(mean(angleDraws), Constants::pi / 8, .05)
      << square->logMessages();
}
//---------------------------------------------------------------------------
TEST_F(McmcTest, SquareResizeLogLikelihood) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  Vector logLikelihoods;
  square->setSideLength(15);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(16);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(17);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(18);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(19);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(20);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));
  square->setSideLength(21);
  logLikelihoods.push_back(Object2D::integratedLogLikelihood(
      screen_, BoundingBox(screen_), objects, background_));

  // I'd like to get the side length exactly right, but off by one is probably
  // fine due to pixel effects.
  EXPECT_TRUE(logLikelihoods[2] == max(logLikelihoods) ||
              logLikelihoods[3] == max(logLikelihoods))
      << cbind(seq(15, 21), logLikelihoods - min(logLikelihoods));
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, ResizeSquare) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  square->setSideLength(4);
  for (int i = 0; i < 100; ++i) {
    square->proposeRescale(BOOM::GlobalRng::rng, screen_, BoundingBox(screen_),
                           objects, background_);
  }
  EXPECT_NEAR(square->side(), 18.0, 2.0) << square->logMessages();
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, ResizeSquareWithSlice) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(square);

  square->setSideLength(4);
  for (int i = 0; i < 100; ++i) {
    square->sliceRescale(BOOM::GlobalRng::rng, screen_, BoundingBox(screen_),
                         objects, background_);
  }
  EXPECT_NEAR(square->side(), 18.0, 2.0) << square->logMessages();
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, CircleTranslationLogLikelihood) {
  std::shared_ptr<Circle> circle(new Circle(80, 120, 7, circlePrior_));
  sim_.addCircle(circle);
  sim_.simulate();

  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(circle);
  BoundingBox region(70, 90, 110, 130);
  Vector logLikelihoods;
  logLikelihoods.push_back(
      Object2D::integratedLogLikelihood(screen_, region, objects, background_));
  circle->moveTo(Point2D(79, 120));
  logLikelihoods.push_back(
      Object2D::integratedLogLikelihood(screen_, region, objects, background_));
  circle->moveTo(Point2D(81, 120));
  logLikelihoods.push_back(
      Object2D::integratedLogLikelihood(screen_, region, objects, background_));
  circle->moveTo(Point2D(80, 119));
  logLikelihoods.push_back(
      Object2D::integratedLogLikelihood(screen_, region, objects, background_));
  circle->moveTo(Point2D(80, 121));
  logLikelihoods.push_back(
      Object2D::integratedLogLikelihood(screen_, region, objects, background_));

  EXPECT_DOUBLE_EQ(logLikelihoods[0], max(logLikelihoods));
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, TranslateCircle) {
  std::shared_ptr<Circle> circle(new Circle(80, 120, 12, circlePrior_));
  sim_.addCircle(circle);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(circle);

  int niter = 100;
  for (int i = 0; i < niter; ++i) {
    circle->proposeMove(GlobalRng::rng, screen_, BoundingBox(screen_), objects,
                        background_);
  }
  EXPECT_LT(distance(circle->center(), Point2D(80, 120)), 2.0)
      << "Circle moved away from true value... " << endl
      << "Center should be (80, 120), actual value: " << circle->center()
      << endl
      << circle->logMessages();

  circle->moveTo(Point2D(120, 120));
  circle->clearLogMessages();

  //  screen_.draw("translateCircle.png");

  circle->proposeGridSearch(GlobalRng::rng, screen_, BoundingBox(screen_),
                            objects, background_);

  for (int i = 0; i < niter; ++i) {
    circle->proposeMove(GlobalRng::rng, screen_, BoundingBox(screen_), objects,
                        background_);
  }
  EXPECT_LT(distance(circle->center(), Point2D(80, 120)), 2.0)
      << "Circle failed to find true value." << endl
      << "Center should be (80, 120), actual value: " << circle->center()
      << endl
      << circle->logMessages();
}

//---------------------------------------------------------------------------
TEST_F(McmcTest, RescaleCircle) {
  std::shared_ptr<Circle> circle(new Circle(80, 120, 13, circlePrior_));
  sim_.addCircle(circle);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(circle);

  circle->setRadius(18);
  for (int i = 0; i < 500; ++i) {
    circle->proposeRescale(GlobalRng::rng, screen_, BoundingBox(screen_),
                           objects, background_);
  }
  EXPECT_NEAR(circle->radius(), 13, 2.0) << circle->logMessages();
}
//---------------------------------------------------------------------------
// Check that an object that should be a circle can get changed from a square
// to a circle.
TEST_F(McmcTest, MorphToCircleTest) {
  std::shared_ptr<Circle> circle(new Circle(80, 120, 13, circlePrior_));
  sim_.addCircle(circle);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;
  objects.push_back(circle);

  objects[0] = sampler_.randomMorph(GlobalRng::rng, objects[0]);
  EXPECT_EQ(objects[0]->type(), ObjectType::SQUARE);

  double circle_fraction = 0;
  int niter = 200;
  for (int i = 0; i < niter; ++i) {
    sampler_.morphSingleObject(0, GlobalRng::rng, screen_, BoundingBox(screen_),
                               objects);
    objects[0]->samplePosterior(GlobalRng::rng, screen_, BoundingBox(screen_),
                                objects, background_);
    circle_fraction += objects[0]->type() == ObjectType::CIRCLE;
  }
  circle_fraction /= niter;
  EXPECT_GT(circle_fraction, .8)
      << objects[0]->logMessages()
      << screen_.draw_and_log("mcmc_morph_to_circle_test.png");
}

//---------------------------------------------------------------------------
// Check that an object that should be a square can get changed from a circle to
// a square.
TEST_F(McmcTest, MorphToSquareTest) {
  std::shared_ptr<Square> square(
      new Square(80, 120, 17, Constants::pi / 3, squarePrior_));
  sim_.addSquare(square);
  sim_.simulate();
  std::vector<std::shared_ptr<Object2D>> objects;

  objects.emplace_back(
      new Circle(80, 120, 17 / sqrt(Constants::pi), circlePrior_));
  double square_fraction = 0;
  double centerMean_x = 0;
  double centerMean_y = 0;
  int niter = 100;
  for (int i = 0; i < niter; ++i) {
    sampler_.morphSingleObject(0, GlobalRng::rng, screen_, BoundingBox(screen_),
                               objects);
    objects[0]->samplePosterior(GlobalRng::rng, screen_, BoundingBox(screen_),
                                objects, background_);
    square_fraction += objects[0]->type() == ObjectType::SQUARE;
    centerMean_x += objects[0]->center().x();
    centerMean_y += objects[0]->center().y();
  }
  square_fraction /= niter;
  centerMean_x /= niter;
  centerMean_y /= niter;
  EXPECT_GT(square_fraction, .8) << "Original object was " << *square << endl
                                 << objects[0]->logMessages();
}

} // namespace
