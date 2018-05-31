#pragma once

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "png++/png.hpp"

namespace slam {
class SimulationCourse;

// A virtual video screen modeling the visual image available to the aircraft
// during a simulated flight.  At any given time a fixed rectangle is
// available.  The bottom of the rectangle starts at pixel 0 and then
// progresses at a fixed speed until the simulation finishes.
//
// The screen maintains a view of (x, y) coordinates.  These are with respect
// to the underlying image to which the screen provides a view.  As the flight
// progresses through space, the bottom frame of the screen does not stay at
// zero.
class Screen2D {
public:
  // Args:
  //   screen_height:  The height of the screen in pixels.
  //   sim: A 2D simulated flight that will be shown on the screen.  This
  //     object does NOT take ownership of sim.  If 'sim' is heap-allocated
  //     then it is the caller's responsibility to make sure it is deleted.
  Screen2D(int screen_height, SimulationCourse *sim);

  // Advance the screen forward 'speed' pixels.
  void advance(int speed);

  std::string nextFilename();

  // Generate a png frame showing the image, and write it to a file.
  void draw(const std::string &filename = "");

  std::string draw_and_log(const std::string &filename) {
    draw(filename);
    std::ostringstream out;
    out << "Screen image logged to " << filename;
    return out.str();
  }

  // Returns true if the top of the screen has reached the end of the flight.
  bool done() const;

  // Return map coordinates of the screen boundaries.  These limits are pixel
  // addresses included on the screen.  The top and right limits are part of the
  // screen, rather than one past the end.
  double bottom() const;
  double top() const;
  double left() const;
  double right() const;

  enum Scale { GRAY, LOGIT };
  // Returns the grayscale value shown at pixel (x, y), assuming (x, y) is on
  // the screen.  Returns -infinity if (x, y) is off the screen.
  //
  // Args:
  //   x, y: The horizontal and vertical positions on the screen.  These are
  //     measured from the origin of the flight.  I.e. (0, 0) will not always
  //     be on the screen.
  //   scale: The scale on which the pixel value is desired.  The default is a
  //     grayscale number in [0, 255] with 0 being black and 255 white.  The
  //     value can also be given on the logit scale, so that it can be modeled
  //     as real-valued (without upper and lower constraints).
  //
  // Pixels are discrete grid points, but (x, y) are real valued.  As with
  // doubles and integers, real valued points map to pixels by rounding down,
  // so (0, 0.1) counts as (0, 0), and (1.8, 2.7) counts as (1, 2).
  double operator()(double x, double y, Scale scale = GRAY) const;

private:
  // The number of pixels in the screen's vertical dimension.
  int screenHeightPixels_;

  SimulationCourse *simulation_;

  // The row number in simulation_->pixels() corresponding to the topmost row
  // in the screen.
  int top_;

  // A counter, starting from 0 (obv!) used to make sure all the files
  // produced by the draw() method have different file names.
  int frame_number_;
};

} // namespace slam
