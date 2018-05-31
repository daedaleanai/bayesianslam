#include "exp/user/stb/slam/object_sampler.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"
#include "exp/user/stb/slam/simulation2d.hh"

using namespace BOOM;
using namespace slam;
using std::cout;
using std::endl;

int main(int argc, char **argv) {

  BOOM::GlobalRng::rng.seed(7);

  // Width of the screen, in pixels.  This number of pixels will be also be seen
  // on the height of the screen in a moving window.
  int screenPixelWidth = 200;
  int screenPixelHeight = 200;

  // Length of the flight, in pixels;
  int flightPixelLength = 200;

  std::shared_ptr<CirclePrior> circlePrior(
      new CirclePrior(0.0, new UniformModel(1, 25),
                      new BOOM::GaussianModelGivenSigma(nullptr, 0, .001),
                      new BOOM::ChisqModel(20, 1.0)));
  std::shared_ptr<SquarePrior> squarePrior(new SquarePrior(
      0.0, new UniformModel(1, 25), new UniformModel(0, Constants::pi / 2),
      new BOOM::GaussianModelGivenSigma(nullptr, 0, .001),
      new BOOM::ChisqModel(20, 1.0)));

  NEW(BOOM::ChisqModel, backgroundColorPrecisionPrior)(20, 1.0);
  NEW(BOOM::GaussianModelGivenSigma, backgroundColorMeanPrior)
  (nullptr, .5, .1);
  std::shared_ptr<Background2D> background(new Background2D(
      backgroundColorMeanPrior, backgroundColorPrecisionPrior));

  SimulationCourse sim(flightPixelLength, screenPixelWidth, circlePrior,
                       squarePrior, background);

  std::shared_ptr<Circle> circle(new Circle(40, 40, 10, circlePrior.get()));
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior.get()));
  std::shared_ptr<Circle> circle2(new Circle(50, 60, 12, circlePrior.get()));
  std::shared_ptr<Square> square2(
      new Square(150, 80, 6, Constants::pi / 6, squarePrior.get()));

  sim.addCircle(circle);
  sim.addSquare(square);
  sim.addSquare(square2);
  sim.addCircle(circle2);
  sim.simulate();
  Screen2D screen(screenPixelHeight, &sim);
  screen.draw();

  ObjectSampler sampler(background, 30);
  sampler.addObjectType(circlePrior);
  sampler.addObjectType(squarePrior);

  sampler.scanForNewObjects(BOOM::GlobalRng::rng, screen, BoundingBox(screen),
                            1000);

} // main
