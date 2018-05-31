#include <iomanip>

#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"
#include "exp/user/stb/slam/simulation2d.hh"

#include "png++/png.hpp"

namespace slam {

Screen2D::Screen2D(int screenHeight, SimulationCourse *sim)
    : screenHeightPixels_(screenHeight), simulation_(sim),
      top_(
          std::max(0, simulation_->flightLengthPixels() - screenHeightPixels_)),
      frame_number_(0) {}

void Screen2D::advance(int speed) {
  if (done()) {
    return;
  }
  if (speed > 0) {
    // Shift the data matrix down by 'speed' rows.
    top_ -= speed;
    if (top_ < 0) {
      top_ = 0;
    }
  }
}

bool Screen2D::done() const { return top_ == 0; }

double Screen2D::bottom() const {
  return std::max(top() - screenHeightPixels_ + 1, 0.0);
}
double Screen2D::top() const {
  return std::max(0, simulation_->flightLengthPixels() - 1 - top_);
}
double Screen2D::left() const { return 0; }
double Screen2D::right() const { return simulation_->screenWidthPixels() - 1; }

double Screen2D::operator()(double x, double y, Scale scale) const {
  x = floor(x);
  y = floor(y);
  if (x < left() || x > right() || y < bottom() || y > top()) {
    return BOOM::negative_infinity();
  }
  std::pair<int, int> px = simulation_->xy_toPixels(x, y);
  if (px.first < 0 || px.first >= simulation_->flightLengthPixels()) {
    std::ostringstream err;
    err << "First pixel coordinate out of bounds."
        << "x = " << x << " y = " << y << " i = " << px.first
        << " j = " << px.second;
    BOOM::report_error(err.str());
  }
  if (px.second < 0 || px.second >= simulation_->screenWidthPixels()) {
    std::ostringstream err;
    err << "Second pixel coordinate out of bounds."
        << "x = " << x << " y = " << y << " i = " << px.first
        << " j = " << px.second;
    BOOM::report_error(err.str());
  }

  double ans = simulation_->pixels()(px.first, px.second);
  switch (scale) {
  case GRAY: {
    return ans;
  }
  case LOGIT: {
    return BOOM::qlogis((ans + 1) / 257);
  }
  default: {
    BOOM::report_error("Unknown scale.");
    return BOOM::negative_infinity();
  }
  }
}

namespace {
inline int truncateToRange(int x, int lo, int hi) {
  if (x < lo)
    x = lo;
  return x > hi ? hi : x;
}
} // namespace

std::string Screen2D::nextFilename() {
  std::ostringstream fname;
  fname << "screen_image_" << std::setw(8) << std::setfill('0')
        << frame_number_++ << ".png";
  return fname.str();
}

void Screen2D::draw(const std::string &filename) {
  int height = simulation_->flightLengthPixels();
  int width = simulation_->screenWidthPixels();

  int bottom = truncateToRange(lround(this->bottom()), 0, height - 1);
  int top = truncateToRange(lround(this->top()), 0, height - 1);
  int left = truncateToRange(lround(this->left()), 0, width - 1);
  int right = truncateToRange(lround(this->right()), 0, width - 1);
  BOOM::ConstSubMatrix data(simulation_->pixels(), bottom, top, left, right);

  png::image<png::gray_pixel> image(data.ncol(), data.nrow());
  for (int i = 0; i < data.nrow(); ++i) {
    for (int j = 0; j < data.ncol(); ++j) {
      png::gray_pixel pixel(static_cast<int>(data(i, j)));
      image.set_pixel(j, i, pixel);
    }
  }
  image.write(filename == "" ? nextFilename() : filename);
}

} // namespace slam
