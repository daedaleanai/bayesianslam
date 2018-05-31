#include "exp/user/stb/slam/aircraft.hh"
#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"
#include "exp/user/stb/slam/simulation2d.hh"

using namespace slam;
using std::cout;
using std::endl;

// A 2d simulation where the aircraft flies straight and finds circles and
// squares on white-noise background.
int main(int argc, char **argv) {

  // Width of the screen, in pixels.  This number of pixels will be also be seen
  // on the height of the screen in a moving window.
  int screen_pixel_width = 200;
  int screen_pixel_height = 200;

  // Length of the flight, in pixels;
  int flight_pixel_length = 5000;

  // Pixels per frame
  int speed = 10;

  std::shared_ptr<Background2D> background(new Background2D(
      new BOOM::GaussianModelGivenSigma(nullptr, 0, .01);
      new BOOM::ChisqModel(20, 1.0)));

  std::shared_ptr<CirclePrior> circlePrior(new CirclePrior(
      30.0 / flight_pixel_length, new UniformModel(1, 25),
      new GaussianModelGivenSigma(nullptr, 0, .0001), new ChisqModel(20, .01)));

  std::shared_ptr<SquarePrior> squarePrior(new SquarePrior(
      30.0 / flight_pixel_length, new UniformModel(1, 25),
      new UniformModel(0, BOOM::Constants::pi / 2),
      new GaussianModelGivenSigma(nullptr, 0, .0001), new ChisqModel(20, .01)));

  SimulationCourse sim(flight_pixel_length, screen_pixel_width, circlePrior,
                       squarePrior, background);
  sim.simulate();
  Screen2D screen(screen_pixel_height, &sim);
  //  AircraftPose2d aircraft(0, screen_pixel_width / 2, 0, speed);

  while (!screen.done()) {
    screen.draw();
    //    aircraft.update_belief(screen);
    screen.advance(speed);
  }

} // main
