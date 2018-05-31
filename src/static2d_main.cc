#include "exp/user/stb/slam/aircraft.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"
#include "exp/user/stb/slam/simulation2d.hh"

using namespace slam;
using namespace BOOM;
using std::cout;
using std::endl;

// A 2d simulation where the aircraft flies straight and finds circles and
// squares on white-noise background.
int main(int argc, char **argv) {

  BOOM::GlobalRng::rng.seed(7);

  // Width of the screen, in pixels.  This number of pixels will be also be seen
  // on the height of the screen in a moving window.
  int screen_pixel_width = 200;
  int screen_pixel_height = 200;

  // Length of the flight, in pixels;
  int flight_pixel_length = 200;

  std::shared_ptr<Background2D> background(new Background2D(
      new GaussianModelGivenSigma(nullptr, .5, .1), new ChisqModel(20, 1.0)));
  std::shared_ptr<CirclePrior> circlePrior(new CirclePrior(
      0.0 / flight_pixel_length, new UniformModel(1, 25),
      new GaussianModelGivenSigma(nullptr, 0, .001), new ChisqModel(20, 0.01)));
  std::shared_ptr<SquarePrior> squarePrior(new SquarePrior(
      0.0 / flight_pixel_length, new UniformModel(1, 25),
      new UniformModel(0, BOOM::Constants::pi / 2),
      new GaussianModelGivenSigma(nullptr, 0, .001), new ChisqModel(20, 0.01)));

  SimulationCourse sim(flight_pixel_length, screen_pixel_width, circlePrior,
                       squarePrior, background);
  std::shared_ptr<Circle> circle(new Circle(40, 40, 10, circlePrior));
  std::shared_ptr<Square> square(
      new Square(80, 120, 18, Constants::pi / 8, squarePrior));
  std::shared_ptr<Circle> circle2(new Circle(50, 60, 12, circlePrior));
  std::shared_ptr<Square> square2(
      new Square(150, 80, 6, Constants::pi / 6, squarePrior));

  sim.addCircle(circle);
  sim.addSquare(square);
  sim.addSquare(square2);
  sim.addCircle(circle2);
  sim.simulate();
  Screen2D screen(screen_pixel_height, &sim);
  screen.draw();

  std::shared_ptr<ObjectSampler> sampler(new ObjectSampler(background, 30));

  Aircraft2D aircraft(1000, background, sampler);
  std::vector<std::shared_ptr<Object2D>> starting_values;

  starting_values.emplace_back(circle->clone());
  starting_values.emplace_back(square->clone());
  starting_values.emplace_back(circle2->clone());
  starting_values.emplace_back(square2->clone());

  aircraft.objectSampler()->scanForNewObjects(BOOM::GlobalRng::rng, screen,
                                              BoundingBox(screen), 1000);

} // main
