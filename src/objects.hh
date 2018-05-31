/** @file objects.hh
 * Classes for modeling two dimensional objects.
 */
#pragma once

#include <iostream>

#include "exp/user/stb/slam/point.hh"
#include "exp/user/stb/slam/region.hh"

#include "cpputil/Constants.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/UniformModel.hpp"
#include "distributions.hpp"

/**
 * @namespace
 * The slam namespace is for the experimental slam algorithm.
 */
namespace slam {
using BOOM::ChisqModel;
using BOOM::GaussianModelGivenSigma;
using BOOM::Ptr;
using BOOM::UniformModel;
using std::cout;
using std::endl;

// Forward declarations
class Screen2D;
class Background2D;
class ObjectPrior2D;

/** @typedef
 * For switching the method used to implement posteior sampling.
 */
enum class PosteriorSamplingMethod { SLICE, MH, MIXED };

/** @typedef
 * A list of legel object types.
 */
enum class ObjectType { BACKGROUND, SQUARE, CIRCLE };

/**
 * A streaming operator for printing object types.
 */
inline std::ostream &operator<<(std::ostream &out, ObjectType type) {
  switch (type) {
  case ObjectType::BACKGROUND: {
    out << "Background";
    break;
  }

  case ObjectType::SQUARE: {
    out << "Square";
    break;
  }

  case ObjectType::CIRCLE: {
    out << "Circle";
    break;
  }

  default: { out << "Unknown"; }
  }
  return out;
}

/** @class Rotation
 * A functor for rotating points at an arbitrary angle around an arbitrary
 * center.
 */
class Rotation {
public:
  Rotation(double angle, const Point2D &center)
      : center_(center), cosine_(cos(angle)), sine_(sin(angle)) {}

  Point2D operator()(const Point2D &point) const {
    double x = point.x() - center_.x();
    double y = point.y() - center_.y();

    // cos -sin   *   x
    // sin  cos       y
    return Point2D(cosine_ * x - sine_ * y + center_.x(),
                   sine_ * x + cosine_ * y + center_.y());
  }

private:
  Point2D center_;
  double cosine_;
  double sine_;
};

/** @class Object2D
 * A base class for modeling 2D objects.
 */
class Object2D {
public:
  Object2D(const Ptr<GaussianModelGivenSigma> &colorMeanPrior,
           const Ptr<ChisqModel> &colorPrecisionPrior);

  static int likelihoodVerbosity;
  static void setLikelihoodVerbosity(int value) { likelihoodVerbosity = value; }

  virtual ~Object2D();

  virtual Object2D *clone() const = 0;

  // Return a box containing the entire object (and maybe more).  This need
  // not be a minimal bounding box.
  virtual BoundingBox boundingBox() const = 0;

  // Check whether the argument is contained in the object.
  virtual bool contains(const Point2D &point) const = 0;

  virtual std::ostream &print(std::ostream &out) const = 0;
  std::string toString() const;
  virtual ObjectType type() const = 0;

  // Returns the real-valued area occupied by the object.
  virtual double area() const = 0;

  // Resize the object so that its area matches the given argument.
  virtual void scaleToArea(double area) = 0;

  virtual Point2D center() const = 0;
  virtual void moveTo(const Point2D &center) = 0;
  virtual void scaleToBoundingBoxVolume(double volume) = 0;

  void addLogMessage(const std::string &message);
  std::string logMessages() const;
  void clearLogMessages() { logMessages_.clear(); }

  double colorMean() const { return colorMean_; }
  void setColorMean(double colorMean);

  double colorSd() const { return colorSd_; }
  void setColorSd(double colorSd);

  // Simulates a color value between 0 (black) and 255 (white).
  int simulateColor(BOOM::RNG &rng) const;

  Ptr<GaussianModelGivenSigma> colorMeanPrior() const {
    return colorMeanPrior_;
  }

  Ptr<ChisqModel> colorPrecisionPrior() const { return colorPrecisionPrior_; }

  // Sampler the size and shape parameters of the object.
  virtual void
  samplePosterior(BOOM::RNG &rng, const Screen2D &screen,
                  const BoundingBox &region,
                  const std::vector<std::shared_ptr<Object2D>> &objects,
                  const std::shared_ptr<Background2D> &background) = 0;

  void drawColorsGivenSuf(BOOM::RNG &rng, const BOOM::GaussianSuf &suf);
  void drawColors(BOOM::RNG &rng, const Screen2D &screen);

  // Returns the log likelihood of a screen subsection after integrating out
  // the color distribution of each object.
  //
  // Args:
  //   screen:  The screen containing the image to be evaluated.
  //   box:  A bounding box giving a subset of the screen to be evaluated.
  //   objects: A collection of objects that may or may not appear on the
  //     screen.  The order is important.  Objects early in the sequence
  //     obscure objects later in the sequence.
  //   background:  The model for the background noise.
  static double
  integratedLogLikelihood(const Screen2D &screen, const Region &box,
                          const std::vector<std::shared_ptr<Object2D>> &objects,
                          const std::shared_ptr<Background2D> &background);

  // Attempt to move the object to a new location int its bounding box.
  // Args:
  //   rng:  The random number generator.
  //   screen:  The screen containing the image.
  //   region:  A bounding box limiting the proposal to a subset of the screen.
  //   objects:  The collection of all objects on the screen (including this
  //   one). background:  The background object.
  //
  // Effects:
  //   A new location for this object is chosen uniformly at random from
  //   'region'.  The MH ratio is computed, and the proposal is stochastically
  //   accepted or rejected.  If accepted the object center will have a new
  //   value.
  void proposeMove(BOOM::RNG &rng, const Screen2D &screen,
                   const BoundingBox &region,
                   const std::vector<std::shared_ptr<Object2D>> &objects,
                   const std::shared_ptr<Background2D> &background);

  // A very expensive MH move that can be used to initialize an object.  The
  // proposal region is divided into grid cells each the size of the object's
  // current bounding box.  Each cell is assigned a sampling weight
  // proportional to the likelihood obtained by moving the object to the
  // center of the cell.
  //
  // The MH proposal is to select a cell at random using these sampling
  // weights, then choose a location within the cell uniformly randomly.
  //
  // Args:
  //   rng:  The random number generator.
  //   screen:  The screen containing the image.
  //   region:  A bounding box limiting the proposal to a subset of the screen.
  //   objects:  The collection of all objects on the screen (including this
  //   one). background:  The background object.
  //
  // Effects:
  //   If the proposal is accepted then the object's center shifts to the
  //   proposed location.
  void proposeGridSearch(BOOM::RNG &rng, const Screen2D &screen,
                         const BoundingBox &region,
                         const std::vector<std::shared_ptr<Object2D>> &objects,
                         const std::shared_ptr<Background2D> &background);

private:
  double colorMean_;
  double colorSd_;
  Ptr<GaussianModelGivenSigma> colorMeanPrior_;
  Ptr<ChisqModel> colorPrecisionPrior_;
  std::vector<std::string> logMessages_;
};

/**
 * Objects of type Object2D can be streamed generically.
 */
std::ostream &operator<<(std::ostream &out, const Object2D &object);

/** @class ObjectPrior2D
 * Interface for prior distributions on 2D objects. Each concrete Object2D
 * class has a corresponding prior distribution class inheriting from
 * ObjectPrior2D.
 */
class ObjectPrior2D {
public:
  virtual ~ObjectPrior2D() {}

  virtual ObjectType type() const = 0;

  // The object's rate of occurance per unit area.  This is the 'lambda'
  // parameter of a homogeneous Poisson process.
  virtual double intensity() const = 0;

  // Args:
  //   rng:  The random number generator.
  //   region: The returned object will be located uniformly at random from
  //     this region.
  //
  // Returns:
  //   A newly allocated object of the type managed by the concrete class.
  virtual std::shared_ptr<Object2D>
  simulate(BOOM::RNG &rng, const BoundingBox &region) const = 0;

  virtual BOOM::Ptr<BOOM::GaussianModelGivenSigma> colorMeanPrior() const = 0;
  virtual BOOM::Ptr<BOOM::ChisqModel> colorPrecisionPrior() const = 0;

  virtual double logpri(const Object2D &object) const = 0;
};

/** @class Background2D
 * The default object to which a pixel will be assigned when it is not
 * contained within another object.
 */
class Background2D : public Object2D {
public:
  // Mean and standard deviation of the logit of the grayscale values.  These
  // numbers get converted to grayscale by z ~ N(mean, sd), with
  // gray = floor(256 * exp(z)/(1 + exp(z))).
  Background2D(const Ptr<GaussianModelGivenSigma> &colorMeanPrior,
               const Ptr<ChisqModel> &colorPrecisionPrior)
      : Object2D(colorMeanPrior, colorPrecisionPrior) {}

  BoundingBox boundingBox() const override { return BoundingBox(); }

  Background2D *clone() const override { return new Background2D(*this); }

  bool contains(const Point2D &point) const override { return true; }

  double area() const override { return BOOM::infinity(); }
  void scaleToArea(double area) override {}

  std::ostream &print(std::ostream &out) const override {
    out << "Background: color mean: " << colorMean()
        << " color sd: " << colorSd();
    return out;
  }

  ObjectType type() const override { return ObjectType::BACKGROUND; }

  Point2D center() const override { return Point2D(-1, -1); }
  void moveTo(const Point2D &) override {}
  void scaleToBoundingBoxVolume(double) override {}

  void
  samplePosterior(BOOM::RNG &rng, const Screen2D &screen,
                  const BoundingBox &region,
                  const std::vector<std::shared_ptr<Object2D>> &objects,
                  const std::shared_ptr<Background2D> &background) override;
};

class SquarePrior;
/** @class
 * A square object.
 */
class Square : public Object2D {
public:
  // Args:
  //   x, y:  Location of the center;
  //   side:  Length of a side (must be positive).
  //   angle:  Between 0 and 2pi;
  //   color:  Grayscale, between 0 and 1;
  Square(double x, double y, double side, double angle, SquarePrior *prior);
  Square(double x, double y, double side, double angle,
         const std::shared_ptr<SquarePrior> &prior)
      : Square(x, y, side, angle, prior.get()) {}

  Square *clone() const override { return new Square(*this); }

  // This bounding box is non-minimal, but it avoids doing a bunch of trig.
  BoundingBox boundingBox() const override {
    double radius = sideLength_ / BOOM::Constants::root2;
    return BoundingBox(x_ - radius, x_ + radius, y_ - radius, y_ + radius);
  }

  double x() const { return x_; }
  double y() const { return y_; }
  double side() const { return sideLength_; }
  double angle() const { return angle_; }

  double area() const override { return BOOM::square(sideLength_); }

  void setSideLength(double sideLength);
  void setAngle(double angle);

  // Maps an angle that might be outside [0, pi / 2) to the range [0, pi / 2).
  double normalizeAngle(double angle);

  Point2D center() const override { return Point2D(x_, y_); }
  void moveTo(const Point2D &center) override {
    x_ = center.x();
    y_ = center.y();
  }
  void scaleToBoundingBoxVolume(double volume) override;
  void scaleToArea(double area) override;

  bool contains(const Point2D &point) const override {
    Rotation unrotate(-angle_, Point2D(x_, y_));
    Point2D aligned = unrotate(point);
    return aligned.x() >= left() && aligned.x() <= right() &&
           aligned.y() <= top() && aligned.y() >= bottom();
  }

  std::ostream &print(std::ostream &out) const override {
    out << "Square: "
        << "center = (" << x_ << ", " << y_ << ") side = " << sideLength_
        << " angle = " << angle_;
    return out;
  }

  ObjectType type() const override { return ObjectType::SQUARE; }

  void
  samplePosterior(BOOM::RNG &rng, const Screen2D &screen,
                  const BoundingBox &region,
                  const std::vector<std::shared_ptr<Object2D>> &objects,
                  const std::shared_ptr<Background2D> &background) override;

  void
  samplePosteriorSlice(BOOM::RNG &rng, const Screen2D &screen,
                       const BoundingBox &region,
                       const std::vector<std::shared_ptr<Object2D>> &objects,
                       const std::shared_ptr<Background2D> &background);

  void proposeRescale(BOOM::RNG &rng, const Screen2D &screen,
                      const BoundingBox &region,
                      const std::vector<std::shared_ptr<Object2D>> &objects,
                      const std::shared_ptr<Background2D> &background);

  // Change the size length parameter using the slice sampler, keeping all
  // other parameters the same.
  void sliceRescale(BOOM::RNG &rng, const Screen2D &screen,
                    const BoundingBox &region,
                    const std::vector<std::shared_ptr<Object2D>> &objects,
                    const std::shared_ptr<Background2D> &background);

  void proposeRotate(BOOM::RNG &rng, const Screen2D &screen,
                     const BoundingBox &region,
                     const std::vector<std::shared_ptr<Object2D>> &objects,
                     const std::shared_ptr<Background2D> &background);

  void sliceRotate(BOOM::RNG &rng, const Screen2D &screen,
                   const BoundingBox &region,
                   const std::vector<std::shared_ptr<Object2D>> &objects,
                   const std::shared_ptr<Background2D> &background);

private:
  // Location of the center, in pixels.
  double x_;
  double y_;
  double sideLength_; // > 0
  double angle_;      // between 0 and pi / 2;

  SquarePrior *prior_;
  PosteriorSamplingMethod posteriorSamplingMethod_;

  // The boundaries of the unrotated square.
  double left() const { return x() - sideLength_ / 2.0; }
  double right() const { return x() + sideLength_ / 2.0; }
  double top() const { return y() + sideLength_ / 2.0; }
  double bottom() const { return y() - sideLength_ / 2.0; }
};

//===========================================================================
/** @class SquarePrior
 * Prior distribution for Square objects.
 */
class SquarePrior : public ObjectPrior2D {
public:
  SquarePrior(double intensity, const Ptr<BOOM::UniformModel> &sideLength,
              const Ptr<BOOM::UniformModel> &angle,
              const Ptr<GaussianModelGivenSigma> &colorMean,
              const Ptr<ChisqModel> &color_precision);

  std::shared_ptr<Square> simulateSquare(BOOM::RNG &rng,
                                         const BoundingBox &region) const;
  std::shared_ptr<Object2D> simulate(BOOM::RNG &rng,
                                     const BoundingBox &region) const override {
    return simulateSquare(rng, region);
  }

  Ptr<GaussianModelGivenSigma> colorMeanPrior() const override {
    return colorMeanPrior_;
  }

  Ptr<ChisqModel> colorPrecisionPrior() const override {
    return colorPrecisionPrior_;
  }

  ObjectType type() const override { return ObjectType::SQUARE; }

  double logpri(const Object2D &object) const override;

  double intensity() const override { return intensity_; }
  const BOOM::UniformModel *sideLengthPrior() const {
    return sideLengthPrior_.get();
  }
  const BOOM::UniformModel *anglePrior() const { return anglePrior_.get(); }

private:
  // The Poisson intensity parameter describing the rate of squares per unit
  // area.
  double intensity_;

  Ptr<BOOM::UniformModel> sideLengthPrior_;
  Ptr<BOOM::UniformModel> anglePrior_;

  Ptr<GaussianModelGivenSigma> colorMeanPrior_;
  Ptr<ChisqModel> colorPrecisionPrior_;
};

//===========================================================================
class CirclePrior;
/** @class Circle
 * Models circle objects.
 */
class Circle : public Object2D {
public:
  // Args:
  //   x, y:  Coordinates of the center.
  //   radius:  Radius of the circle.
  //   colorMean: Mean of the color distribution inside the circle, on the
  //     logit scale.  After the inverse logit transformation, a value of 1
  //     corresponds to color 255 (white) and a value of 0 to 0 (black).
  //   colorSd: Standard deviation of the color distribution inside the
  //     circle, on the logit scale.
  Circle(double x, double y, double radius, CirclePrior *prior);
  Circle(double x, double y, double radius,
         const std::shared_ptr<CirclePrior> &prior)
      : Circle(x, y, radius, prior.get()) {}

  Circle *clone() const override { return new Circle(*this); }

  double x() const { return x_; }
  double y() const { return y_; }
  double radius() const { return radius_; }
  void setRadius(double radius);

  Point2D center() const override { return Point2D(x_, y_); }
  void moveTo(const Point2D &center) override {
    x_ = center.x();
    y_ = center.y();
  }

  double area() const override {
    return BOOM::Constants::pi * BOOM::square(radius_);
  }
  void scaleToArea(double area) override;

  void scaleToBoundingBoxVolume(double volume) override;

  BoundingBox boundingBox() const override {
    return BoundingBox(x_ - radius_, x_ + radius_, y_ - radius_, y_ + radius_);
  }

  bool contains(const Point2D &point) const override {
    double dx = point.x() - x_;
    double dy = point.y() - y_;
    return BOOM::square(dx) + BOOM::square(dy) <= BOOM::square(radius_);
  }

  std::ostream &print(std::ostream &out) const override {
    out << "Circle: "
        << "center = (" << x_ << ", " << y_ << ") radius = " << radius_;
    return out;
  }

  ObjectType type() const override { return ObjectType::CIRCLE; }

  void
  samplePosterior(BOOM::RNG &rng, const Screen2D &screen,
                  const BoundingBox &region,
                  const std::vector<std::shared_ptr<Object2D>> &objects,
                  const std::shared_ptr<Background2D> &background) override;

  void
  samplePosteriorSlice(BOOM::RNG &rng, const Screen2D &screen,
                       const BoundingBox &region,
                       const std::vector<std::shared_ptr<Object2D>> &objects,
                       const std::shared_ptr<Background2D> &background);

  void proposeRescale(BOOM::RNG &rng, const Screen2D &screen,
                      const BoundingBox &region,
                      const std::vector<std::shared_ptr<Object2D>> &objects,
                      const std::shared_ptr<Background2D> &background);

private:
  double x_;
  double y_;
  double radius_;
  CirclePrior *prior_;
  PosteriorSamplingMethod posteriorSamplingMethod_;
};

/** @class CirclePrior
 * Prior distribution for circle objects.
 */
class CirclePrior : public ObjectPrior2D {
public:
  CirclePrior(double intensity, const Ptr<BOOM::UniformModel> &radius,
              const Ptr<GaussianModelGivenSigma> &colorMean,
              const Ptr<ChisqModel> &color_precision);
  std::shared_ptr<Circle> simulateCircle(BOOM::RNG &rng,
                                         const BoundingBox &region) const;
  std::shared_ptr<Object2D> simulate(BOOM::RNG &rng,
                                     const BoundingBox &region) const override {
    return simulateCircle(rng, region);
  }
  ObjectType type() const override { return ObjectType::CIRCLE; }
  double logpri(const Object2D &object) const override;
  double intensity() const override { return intensity_; }
  const BOOM::UniformModel *radiusPrior() const { return radiusPrior_.get(); }

  Ptr<GaussianModelGivenSigma> colorMeanPrior() const override {
    return colorMeanPrior_;
  }

  Ptr<ChisqModel> colorPrecisionPrior() const override {
    return colorPrecisionPrior_;
  }

private:
  // The Poisson intensity parameter rate of circles per unit area.
  double intensity_;
  Ptr<BOOM::UniformModel> radiusPrior_;
  Ptr<GaussianModelGivenSigma> colorMeanPrior_;
  Ptr<ChisqModel> colorPrecisionPrior_;
};

} // namespace slam
