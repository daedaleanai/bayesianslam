#pragma once

#include <vector>

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "exp/user/stb/slam/objects.hh"

namespace slam {

// This class generates the ground truth for a 2d simulated flight.
class SimulationCourse {
public:
  SimulationCourse(int flightLengthPixels, int screenWidthPixels,
                   const std::shared_ptr<CirclePrior> &circlePrior,
                   const std::shared_ptr<SquarePrior> &squarePrior,
                   const std::shared_ptr<Background2D> &background);

  // The object will not take ownership of rng.  It is up to the caller to
  // delete it if necessary.
  void setRng(BOOM::RNG *rng) { rng_ = rng; }

  int screenWidthPixels() const { return screenWidthPixels_; }

  // Simulate the population of squares and circles.
  void simulate();

  // Show the next row of pixels, if there is one.  If you've run out of
  // pixels then show the final row.
  BOOM::Vector next_line();

  int flightLengthPixels() const { return flightLengthPixels_; }

  Circle simulateCircle(BOOM::RNG &rng);
  Square simulateSquare(BOOM::RNG &rng);
  void draw(const Object2D &object);

  // Add a pre-defined circle.
  void addCircle(const std::shared_ptr<Circle> &circle) {
    circles_.push_back(circle);
  }

  // Add a pre-defined square.
  void addSquare(const std::shared_ptr<Square> &square) {
    squares_.push_back(square);
  }

  const BOOM::Matrix &pixels() const { return pixels_; }

  // Convert real valued x,y coordinates to the corresponding pixel row and
  // column numbers.  The bottom, left pixel is (0, 0).
  std::pair<int, int> xy_toPixels(double x, double y) const;

private:
  int flightLengthPixels_;
  int screenWidthPixels_;

  std::shared_ptr<CirclePrior> circlePrior_;
  std::shared_ptr<SquarePrior> squarePrior_;
  std::shared_ptr<Background2D> background_;

  std::vector<std::shared_ptr<Square>> squares_;
  std::vector<std::shared_ptr<Circle>> circles_;

  BOOM::Matrix pixels_;

  int nextRowToShow_;

  // A random number generator to use for the simulation.
  BOOM::RNG *rng_;
};

} // namespace slam
