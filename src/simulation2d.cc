#include "exp/user/stb/slam/simulation2d.hh"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"
#include "exp/user/stb/slam/screen2d.hh"

#include "png++/png.hpp"

namespace slam {
using namespace BOOM;

SimulationCourse::SimulationCourse(
    int flightLengthPixels, int screenWidthPixels,
    const std::shared_ptr<CirclePrior> &circlePrior,
    const std::shared_ptr<SquarePrior> &squarePrior,
    const std::shared_ptr<Background2D> &background)
    : flightLengthPixels_(flightLengthPixels),
      screenWidthPixels_(screenWidthPixels), circlePrior_(circlePrior),
      squarePrior_(squarePrior), background_(background),
      nextRowToShow_(flightLengthPixels - 1), rng_(&BOOM::GlobalRng::rng) {}

void SimulationCourse::simulate() {
  int number_of_squares =
      rpois_mt(*rng_, squarePrior_->intensity() * flightLengthPixels_);
  int number_ofCircles =
      rpois_mt(*rng_, circlePrior_->intensity() * flightLengthPixels_);

  pixels_.resize(flightLengthPixels_, screenWidthPixels_);
  for (int i = 0; i < flightLengthPixels_; ++i) {
    for (int j = 0; j < screenWidthPixels_; ++j) {
      int backgroundColor = background_->simulateColor(*rng_);
      pixels_(i, j) = backgroundColor;
    }
  }

  BoundingBox simulationRegion(0, screenWidthPixels_ - 1, 0,
                               flightLengthPixels_ - 1);

  for (int i = 0; i < number_of_squares; ++i) {
    squares_.push_back(squarePrior_->simulateSquare(*rng_, simulationRegion));
  }
  for (int i = 0; i < number_ofCircles; ++i) {
    circles_.push_back(circlePrior_->simulateCircle(*rng_, simulationRegion));
  }

  // By separating the simulating from the drawing we can make sure we include
  // any manually added circles or squares get drawn.

  for (int i = 0; i < circles_.size(); ++i) {
    draw(*circles_[i]);
  }
  for (int i = 0; i < squares_.size(); ++i) {
    draw(*squares_[i]);
  }
}

BOOM::Vector SimulationCourse::next_line() {
  if (pixels_.nrow() == 0) {
    simulate();
  }
  if (nextRowToShow_ > 0) {
    return (pixels_.row(nextRowToShow_--));
  } else {
    return pixels_.row(0);
  }
}

void SimulationCourse::draw(const Object2D &object) {
  BoundingBox box = object.boundingBox();
  for (double x = box.left(); x <= box.right(); ++x) {
    for (double y = box.bottom(); y < box.top(); ++y) {
      if (object.contains(Point2D(x, y))) {
        std::pair<int, int> px = xy_toPixels(x, y);
        if (px.first >= 0 && px.second >= 0) {
          pixels_(px.first, px.second) = object.simulateColor(*rng_);
        }
      }
    }
  }
}

std::pair<int, int> SimulationCourse::xy_toPixels(double x, double y) const {
  int j = lround(x);
  int i = lround(pixels_.nrow() - 1 - y);
  if (i < 0 || j < 0 || i >= pixels_.nrow() || j >= pixels_.ncol()) {
    return std::pair<int, int>(-1, -1);
  }
  return std::make_pair(i, j);
}

} // namespace slam
