#include <cmath>
#include <map>

#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"

#include "Samplers/ScalarSliceSampler.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"
#include "cpputil/ParamHolder.hpp"
#include "distributions.hpp"

namespace slam {
using namespace BOOM;

int Object2D::likelihoodVerbosity = 0;

//===========================================================================
Object2D::Object2D(const Ptr<GaussianModelGivenSigma> &colorMeanPrior,
                   const Ptr<ChisqModel> &colorPrecisionPrior)
    : colorMeanPrior_(colorMeanPrior),
      colorPrecisionPrior_(colorPrecisionPrior) {
  colorSd_ = 1.0 / sqrt(colorPrecisionPrior_->sim());
  colorMean_ =
      rnorm(colorMeanPrior_->mu(), colorSd_ / sqrt(colorMeanPrior_->kappa()));
}

Object2D::~Object2D() {}

void Object2D::setColorMean(double colorMean) { colorMean_ = colorMean; }

void Object2D::addLogMessage(const std::string &message) {
  logMessages_.push_back(message);
}

std::string Object2D::toString() const {
  std::ostringstream out;
  out << *this;
  return out.str();
}

std::string Object2D::logMessages() const {
  std::ostringstream messages;
  for (const auto &msg : logMessages_) {
    messages << msg << std::endl;
  }
  return messages.str();
}

//---------------------------------------------------------------------------
void Object2D::setColorSd(double colorSd) {
  if (colorSd < 0) {
    report_error("Color standard deviation must be non-negative.");
  }
  colorSd_ = colorSd;
}
//---------------------------------------------------------------------------
int Object2D::simulateColor(BOOM::RNG &rng) const {
  double z = BOOM::rnorm_mt(rng, colorMean_, colorSd_);
  return floor(256 * BOOM::plogis(z));
}
//---------------------------------------------------------------------------
void Object2D::drawColorsGivenSuf(RNG &rng, const GaussianSuf &suf) {
  double ss = colorPrecisionPrior_->sum_of_squares();
  double df = colorPrecisionPrior_->df();
  double n = suf.n();
  double kappa = colorMeanPrior_->kappa();
  double posteriorMean =
      (suf.sum() + kappa * colorMeanPrior_->mu()) / (n + kappa);

  double DF = df + n;
  double SS = ss + suf.centered_sumsq(suf.ybar()) +
              n * square(posteriorMean - suf.ybar()) +
              kappa * square(posteriorMean - colorMeanPrior_->mu());

  colorSd_ = sqrt(1.0 / rgamma_mt(rng, DF / 2, SS / 2));
  colorMean_ = rnorm_mt(rng, posteriorMean, colorSd_ / sqrt(n + kappa));
}
//---------------------------------------------------------------------------
void Object2D::drawColors(RNG &rng, const Screen2D &screen) {
  GaussianSuf suf;
  BoundingBox box = boundingBox().intersection(BoundingBox(screen));
  for (double x = box.left(); x <= box.right(); ++x) {
    for (double y = box.bottom(); y <= box.top(); ++y) {
      if (this->contains(Point2D(x, y))) {
        double observation = screen(x, y, Screen2D::LOGIT);
        suf.update_raw(observation);
      }
    }
  }
  drawColorsGivenSuf(rng, suf);
}

//---------------------------------------------------------------------------
namespace {

void increment_suf(const Screen2D &screen, const Region &region,
                   const std::vector<std::shared_ptr<Object2D>> &objects,
                   std::map<std::shared_ptr<Object2D>, GaussianSuf> &suf,
                   GaussianSuf &background_suf) {
  for (const auto &px : region) {
    double observation = screen(px.x(), px.y(), Screen2D::LOGIT);
    if (!std::isfinite(observation)) {
      continue;
    }
    bool found = false;
    for (const auto &objectptr : objects) {
      if (objectptr->contains(px)) {
        found = true;
        suf[objectptr].update_raw(observation);
        if (Object2D::likelihoodVerbosity > 0) {
          cout << px.x() << " " << px.y() << " " << objectptr->type() << " "
               << observation << endl;
        }
        break;
      }
    }
    if (!found) {
      background_suf.update_raw(observation);
      if (Object2D::likelihoodVerbosity > 0) {
        cout << px.x() << " " << px.y() << " BG " << observation << endl;
      }
    }
  }
}

std::vector<std::shared_ptr<Object2D>>
active_objects(const std::vector<std::shared_ptr<Object2D>> &objects,
               const Region &region) {
  std::vector<std::shared_ptr<Object2D>> active;
  for (const auto &obj : objects) {
    if (obj->boundingBox().intersects(region, true)) {
      active.push_back(obj);
    }
  }
  return active;
}

template <class CONTAINER>
double
integratedLogLikelihood_impl(const Screen2D &screen, const BoundingBox &box,
                             const CONTAINER &objects,
                             const std::shared_ptr<Background2D> &background) {
  // Find the portion of 'box' that is actually on the screen.
  const BoundingBox visible_box(BoundingBox(screen).intersection(box));
  if (visible_box.empty()) {
    return 0;
  }

  // Locate the set of objects whose bounding box intersects box.
  std::vector<std::shared_ptr<Object2D>> candidates =
      active_objects(objects, visible_box);

  std::map<std::shared_ptr<Object2D>, GaussianSuf> suf;
  GaussianSuf background_suf;
  for (int x = visible_box.left(); x <= visible_box.right(); ++x) {
    for (double y = visible_box.bottom(); y <= visible_box.top(); ++y) {
      double observation;
      try {
        observation = screen(x, y, Screen2D::LOGIT);
      } catch (std::exception &e) {
        std::ostringstream err;
        err << "caught an exception while evaluating integratedLogLikelihood"
            << endl
            << "error message: " << e.what() << endl
            << "original bounding box: " << box << endl
            << "screen bounding box: " << BoundingBox(screen) << endl
            << "Intersection: " << visible_box << endl;
        report_error(err.str());
      }
      bool found = false;
      for (const auto &objectptr : candidates) {
        if (objectptr->contains(Point2D(x, y))) {
          found = true;
          suf[objectptr].update_raw(observation);
          break;
        }
      }
      if (!found) {
        background_suf.update_raw(observation);
      }
    }
  }

  double log_likelihood = 0;
  for (const auto &el : suf) {
    log_likelihood += GaussianModelBase::log_integrated_likelihood(
        el.second, el.first->colorMeanPrior()->mu(),
        el.first->colorMeanPrior()->kappa(),
        el.first->colorPrecisionPrior()->df(),
        el.first->colorPrecisionPrior()->sum_of_squares());
  }

  log_likelihood += GaussianModelBase::log_integrated_likelihood(
      background_suf, background->colorMeanPrior()->mu(),
      background->colorMeanPrior()->kappa(),
      background->colorPrecisionPrior()->df(),
      background->colorPrecisionPrior()->sum_of_squares());

  return log_likelihood;
}

//-----------------------------------------------------------------
double better_integratedLogLikelihood_impl(
    const Screen2D &screen, const Region &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  // Find the portion of 'region' that is actually on the screen.
  BoxUnion visible_region = intersection(region, BoundingBox(screen));
  if (visible_region.empty()) {
    return 0;
  }
  ImageGrid grid(visible_region.boundingBox(), 100);
  BoxUnion evaluation_region = grid.intersection(visible_region);

  std::map<std::shared_ptr<Object2D>, GaussianSuf> suf;
  GaussianSuf background_suf;
  for (const auto &box : evaluation_region.boxes()) {
    increment_suf(screen, box, active_objects(objects, box), suf,
                  background_suf);
  }

  double log_likelihood = 0;
  if (Object2D::likelihoodVerbosity > 0) {
    cout << "Evaluating log likelihood in : " << region << endl;
  }

  for (const auto &el : suf) {
    if (el.second.n() > 0) {
      double contribution = GaussianModelBase::log_integrated_likelihood(
          el.second, el.first->colorMeanPrior()->mu(),
          el.first->colorMeanPrior()->kappa(),
          el.first->colorPrecisionPrior()->df(),
          el.first->colorPrecisionPrior()->sum_of_squares());

      if (Object2D::likelihoodVerbosity > 0) {
        cout << *el.first << " suf = " << el.second << " " << el.second.n()
             << " " << el.second.ybar() << " " << el.second.sample_var()
             << " loglike: " << contribution << endl;
      }
      log_likelihood += contribution;
      if (std::isnan(log_likelihood)) {
        cout << "found a NaN in log likelihood" << endl;
      }
    }
  }

  if (background_suf.n() > 0) {
    double contribution = GaussianModelBase::log_integrated_likelihood(
        background_suf, background->colorMeanPrior()->mu(),
        background->colorMeanPrior()->kappa(),
        background->colorPrecisionPrior()->df(),
        background->colorPrecisionPrior()->sum_of_squares());
    log_likelihood += contribution;

    if (Object2D::likelihoodVerbosity > 0) {
      cout << "Background suf: " << background_suf
           << " loglike: " << contribution << endl
           << "Total log likelihood: " << log_likelihood << endl;
    }
  }
  return log_likelihood;
}
} // namespace

double Object2D::integratedLogLikelihood(
    const Screen2D &screen, const Region &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  return better_integratedLogLikelihood_impl(screen, region, objects,
                                             background);
}

std::ostream &operator<<(std::ostream &out, const Object2D &obj) {
  return obj.print(out);
}

//===========================================================================
void Background2D::samplePosterior(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  GaussianSuf suf;
  BoundingBox box = region.intersection(BoundingBox(screen));
  for (double x = box.left(); x <= box.right(); ++x) {
    for (double y = box.bottom(); y <= box.top(); ++y) {
      double observation = screen(x, y, Screen2D::LOGIT);
      bool background_pixel = true;
      for (const auto &el : objects) {
        if (el->contains(Point2D(x, y))) {
          background_pixel = false;
          break;
        }
      }
      if (background_pixel) {
        suf.update_raw(observation);
      }
    }
  }
  drawColorsGivenSuf(rng, suf);
}

//===========================================================================
Square::Square(double x, double y, double side, double angle,
               SquarePrior *prior)
    : Object2D(prior->colorMeanPrior(), prior->colorPrecisionPrior()), x_(x),
      y_(y), sideLength_(side), angle_(angle), prior_(prior),
      posteriorSamplingMethod_(PosteriorSamplingMethod::MIXED) {
  if (angle_ < 0 || angle_ > Constants::pi / 2) {
    report_error("Angle angle must be between 0 and pi/2.");
  }
  if (sideLength_ <= 0) {
    report_error("Side length must be positive.");
  }
}

void Square::samplePosterior(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  switch (posteriorSamplingMethod_) {
  case PosteriorSamplingMethod::SLICE: {
    samplePosteriorSlice(rng, screen, region, objects, background);
    break;
  }

  case PosteriorSamplingMethod::MH: {
    proposeMove(rng, screen, region, objects, background);
    proposeRescale(rng, screen, region, objects, background);
    proposeRotate(rng, screen, region, objects, background);
    break;
  }

  case PosteriorSamplingMethod::MIXED: {
    proposeMove(rng, screen, region, objects, background);
    sliceRescale(rng, screen, region, objects, background);
    sliceRotate(rng, screen, region, objects, background);
    break;
  }

  default: {
    report_error("Unknown posterior sampling method when modifying "
                 "Square.");
  }
  }
}

void Square::samplePosteriorSlice(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  Vector params = {x_, y_, sideLength_, angle_};
  std::function<double(const Vector &)> target = [&](const Vector &x) {
    if (x[0] < region.left() || x[0] > region.right() ||
        x[1] < region.bottom() || x[1] > region.top()) {
      return negative_infinity();
    }
    double ans = prior_->sideLengthPrior()->logp(x[2]) +
                 prior_->anglePrior()->logp(x[3]);
    if (!std::isfinite(ans)) {
      return ans;
    }
    double x0 = x_;
    double y0 = y_;
    double s0 = sideLength_;
    double a0 = angle_;

    this->x_ = x[0];
    this->y_ = x[1];
    this->sideLength_ = x[2];
    this->angle_ = x[3];
    ans +=
        Object2D::integratedLogLikelihood(screen, region, objects, background);
    if (!std::isfinite(ans)) {
      std::cout << "Infinity found in Square::samplePosterior." << std::endl;
    }
    x_ = x0;
    y_ = y0;
    sideLength_ = s0;
    angle_ = a0;
    return ans;
  };

  UnivariateSliceSampler sampler(target, 1.0, true, &rng);
  sampler.set_limits(
      {region.left(), region.bottom(), 0, 0},
      {region.right(), region.top(), infinity(), Constants::pi / 2});

  params = sampler.draw(params);
  x_ = params[0];
  y_ = params[1];
  sideLength_ = params[2];
  angle_ = params[3];
  drawColors(rng, screen);
}

void Square::setSideLength(double side) {
  if (side <= 0) {
    report_error("Side length must be positive.");
  }
  sideLength_ = side;
}

double Square::normalizeAngle(double angle) {
  if (angle >= Constants::pi / 2) {
    return std::fmod(angle, Constants::pi / 2);
  } else if (angle < 0) {
    return std::fmod(angle, Constants::pi / 2) + Constants::pi / 2;
  } else {
    return angle;
  }
}

void Square::setAngle(double angle) { angle_ = normalizeAngle(angle); }

void Square::scaleToArea(double area) {
  if (area < 0) {
    report_error("Area can't be negative.");
  }
  sideLength_ = sqrt(area);
}

void Square::scaleToBoundingBoxVolume(double volume) {
  double current_volume = boundingBox().volume();
  double ratio = volume / current_volume;
  sideLength_ *= sqrt(ratio);
}

void Square::proposeRescale(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  double originalSideLength = sideLength_;
  double candidateSideLength;
  if (runif_mt(rng) < .5) {
    candidateSideLength = runif_mt(rng, .5 * sideLength_, 2.0 * sideLength_);
  } else {
    candidateSideLength = runif_mt(rng, sideLength_ * .95, sideLength_ / .95);
  }
  double candidate_logPrior =
      prior_->sideLengthPrior()->logp(candidateSideLength);
  double original_logPrior =
      prior_->sideLengthPrior()->logp(originalSideLength);
  if (!std::isfinite(original_logPrior)) {
    std::cout << logMessages();
    report_error("Illegal starting value for square side length.");
  }
  if (!std::isfinite(candidate_logPrior)) {
    // We have proposed a value that is outside the support of the prior, so
    // reject the proposal.
    return;
  }

  BoundingBox original_bounding_box = boundingBox();
  sideLength_ = candidateSideLength;
  BoundingBox candidate_bounding_box = boundingBox();
  BoundingBox box = candidate_bounding_box.super_box(original_bounding_box);

  double loglike_cand =
      Object2D::integratedLogLikelihood(screen, box, objects, background);
  sideLength_ = originalSideLength;
  double loglike_original =
      Object2D::integratedLogLikelihood(screen, box, objects, background);

  // The proposal distribution is symmetric, so it does not show up here.
  double log_MH_ratio = loglike_cand + candidate_logPrior -
                        (loglike_original + original_logPrior);

  std::ostringstream log_message;
  double log_u = log(runif_mt(rng));
  if (log_u < log_MH_ratio) {
    log_message << "Rescale successful: side length " << originalSideLength
                << " -> " << candidateSideLength << " with log likelihood "
                << loglike_original << " -> " << loglike_cand;
    sideLength_ = candidateSideLength;
  } else {
    log_message << "Rescale failed: side length " << originalSideLength
                << " -/> " << candidateSideLength << " with log likelihood "
                << loglike_original << " -/> " << loglike_cand;
    sideLength_ = originalSideLength;
  }
  addLogMessage(log_message.str());
}

void Square::sliceRescale(BOOM::RNG &rng, const Screen2D &screen,
                          const BoundingBox &region,
                          const std::vector<std::shared_ptr<Object2D>> &objects,
                          const std::shared_ptr<Background2D> &background) {
  std::function<double(double)> log_density = [&](double side) {
    if (side <= 0) {
      return negative_infinity();
    }
    BoundingBox original_bounding_box = boundingBox();
    RealValueHolder s0(this->sideLength_);
    this->sideLength_ = side;
    double logPrior = prior_->sideLengthPrior()->logp(side);
    if (!std::isfinite(logPrior)) {
      return negative_infinity();
    }
    BoundingBox box =
        boundingBox().super_box(original_bounding_box).intersection(region);
    return logPrior +
           Object2D::integratedLogLikelihood(screen, box, objects, background);
  };

  ScalarSliceSampler sampler(log_density, true, 1, &rng);
  sampler.set_lower_limit(0);
  sideLength_ = sampler.draw(sideLength_);
}

void Square::proposeRotate(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  double originalAngle = angle_;
  double candidateAngle;
  if (runif_mt(rng) < .5) {
    // Try a big-ish rotation about half the time.
    candidateAngle =
        runif_mt(rng, angle_ - Constants::pi / 10, angle_ + Constants::pi / 10);
  } else {
    // Try a small rotation up to +/- 2 degrees (about .035 radians) about half
    // the time.
    candidateAngle = runif_mt(rng, angle_ - .035, angle_ + .035);
  }
  candidateAngle = normalizeAngle(candidateAngle);

  double loglike_original =
      Object2D::integratedLogLikelihood(screen, region, objects, background);
  angle_ = candidateAngle;
  double loglike_cand =
      Object2D::integratedLogLikelihood(screen, region, objects, background);
  double log_MH_ratio =
      loglike_cand + prior_->anglePrior()->logp(candidateAngle) -
      (loglike_original + prior_->anglePrior()->logp(originalAngle));
  // The proposal distribution is symmetric, so it disappears from the MH
  // ratio.

  double log_u = log(runif_mt(rng));
  std::ostringstream log_message;
  if (log_u < log_MH_ratio) {
    angle_ = candidateAngle;
    log_message << "Accepted angular move from " << originalAngle << " to "
                << candidateAngle;
  } else {
    angle_ = originalAngle;
    log_message << "Rejected angular move from " << originalAngle << " to "
                << candidateAngle;
  }
  addLogMessage(log_message.str());
}

void Square::sliceRotate(BOOM::RNG &rng, const Screen2D &screen,
                         const BoundingBox &region,
                         const std::vector<std::shared_ptr<Object2D>> &objects,
                         const std::shared_ptr<Background2D> &background) {
  std::function<double(double)> log_density = [&](double angle) {
    angle = normalizeAngle(angle);
    double ans = prior_->anglePrior()->logp(angle);
    if (!std::isfinite(ans)) {
      return negative_infinity();
    }
    RealValueHolder a0(angle_);
    setAngle(angle);
    ans += Object2D::integratedLogLikelihood(screen, this->boundingBox(),
                                             objects, background);
    return ans;
  };
  ScalarSliceSampler sampler(log_density, true, 1, &rng);
  sampler.set_limits(0, BOOM::Constants::pi / 2);
  angle_ = normalizeAngle(sampler.draw(angle_));
}

//===========================================================================

SquarePrior::SquarePrior(double intensity, const Ptr<UniformModel> &sideLength,
                         const Ptr<UniformModel> &angle,
                         const Ptr<GaussianModelGivenSigma> &colorMean,
                         const Ptr<ChisqModel> &color_precision)
    : intensity_(intensity), sideLengthPrior_(sideLength), anglePrior_(angle),
      colorMeanPrior_(colorMean), colorPrecisionPrior_(color_precision) {
  if (intensity_ <= 0) {
    report_error("Intensity must be positive.");
  }
}

std::shared_ptr<Square>
SquarePrior::simulateSquare(RNG &rng, const BoundingBox &region) const {
  double x = runif_mt(rng, region.left(), region.right());
  double y = runif_mt(rng, region.bottom(), region.top());
  double angle = anglePrior_->sim(rng);
  double sideLength = sideLengthPrior_->sim(rng);
  return std::shared_ptr<Square>(
      new Square(x, y, sideLength, angle, const_cast<SquarePrior *>(this)));
}

double SquarePrior::logpri(const Object2D &object) const {
  if (object.type() != ObjectType::SQUARE) {
    report_error("Not a square.");
  }
  const Square &square(static_cast<const Square &>(object));
  return sideLengthPrior_->logp(square.side()) +
         anglePrior_->logp(square.angle());
}

//===========================================================================
Circle::Circle(double x, double y, double radius, CirclePrior *prior)
    : Object2D(prior->colorMeanPrior(), prior->colorPrecisionPrior()), x_(x),
      y_(y), radius_(radius), prior_(prior),
      posteriorSamplingMethod_(PosteriorSamplingMethod::MH) {
  if (radius_ <= 0) {
    BOOM::report_error("Radius must be positive.");
  }
}

void Circle::samplePosterior(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  switch (posteriorSamplingMethod_) {
  case PosteriorSamplingMethod::SLICE: {
    samplePosteriorSlice(rng, screen, region, objects, background);
    break;
  }

  case PosteriorSamplingMethod::MH: {
    proposeMove(rng, screen, region, objects, background);
    proposeRescale(rng, screen, region, objects, background);
    break;
  }

  default: { report_error("Unknown posterior sampling method."); }
  }
}

void Circle::scaleToArea(double area) {
  if (area < 0) {
    report_error("Area can't be negative.");
  }
  radius_ = sqrt(area / Constants::pi);
}

//===========================================================================

// Evaluate the log likelihood on a grid of points determined by the
// simulation region and the bounding box of the object.  The log likelihood
// in each grid cell to randomly choose a grid cell, then sample uniformly
// within the cell.
void Object2D::proposeGridSearch(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  Point2D original_center = center();
  BoundingBox visible_region = region.intersection(BoundingBox(screen));
  BoundingBox original_bounding_box =
      boundingBox().intersection(visible_region);

  // The log likelihood contribution of each grid cell, conditional on the
  // object being centered within the cell.  This will be calculated in two
  // pieces.  First, we calculate the log likelihood contribution of each cell
  // conditional on having the object in the cell center.  Then we move the
  // object to the next cell and compute the cell likelihood without the
  // object present.  The overall log likelihood is log_likelihood[i] +
  // sum(background_log_likelihood[-i], where [-i] means all cells other than
  // i.
  Vector log_likelihood;

  // The log_likelihood contribution of each grid cell, conditional on the
  // object not being in the cell.
  Vector background_log_likelihood;
  std::vector<Point2D> grid_centers;
  int original_index = -1;

  // Shift the object's center so that the lower left corner of the bounding
  // box is in the lower left corner of the visible region.
  double dleft = original_bounding_box.left() - visible_region.left();
  double dbottom = original_bounding_box.bottom() - visible_region.bottom();
  moveTo(Point2D(original_center.x() - dleft, original_center.y() - dbottom));

  BoundingBox current_box = boundingBox();
  while (current_box.right() < visible_region.right()) {
    while (current_box.top() < visible_region.top()) {
      BoundingBox cell = current_box.intersection(visible_region);
      log_likelihood.push_back(
          Object2D::integratedLogLikelihood(screen, cell, objects, background));
      Point2D position = center();
      grid_centers.push_back(position);
      if (current_box.contains(original_center)) {
        original_index = log_likelihood.size() - 1;
      }
      moveTo(
          Point2D(position.x(), position.y() + original_bounding_box.height()));
      // Now that the object has been moved on to the next cell, compute the
      // background log likelihood of the current cell.
      background_log_likelihood.push_back(
          Object2D::integratedLogLikelihood(screen, cell, objects, background));
      current_box = boundingBox();
    }
    Point2D position = center();
    moveTo(Point2D(position.x() + original_bounding_box.width(),
                   visible_region.bottom()));
    current_box = boundingBox();
  }
  // Vector subtraction and then scalar addition.
  log_likelihood -= background_log_likelihood;
  log_likelihood += sum(background_log_likelihood);

  Vector sampling_weights = log_likelihood;
  sampling_weights.normalize_logprob();
  int proposal_index = rmulti_mt(rng, sampling_weights);
  moveTo(grid_centers[proposal_index]);

  BoundingBox box = boundingBox().intersection(visible_region);
  Point2D candidate = Point2D(runif_mt(rng, box.left(), box.right() - 1),
                              runif_mt(rng, box.bottom(), box.top() - 1));
  moveTo(candidate);
  BoxUnion likelihood_region(boundingBox().intersection(visible_region),
                             original_bounding_box);

  double loglike_cand = Object2D::integratedLogLikelihood(
      screen, likelihood_region, objects, background);
  moveTo(original_center);
  double loglike_orig = Object2D::integratedLogLikelihood(
      screen, likelihood_region, objects, background);

  // The proposal distribution is the probability of choosing the given grid
  // cell, divided by the cell's area.  The prior is uniform, and thus ignored
  // here.
  double log_MH_ratio =
      (loglike_cand - log_likelihood[proposal_index] - log(box.area())) -
      (loglike_orig - log_likelihood[original_index] -
       log(original_bounding_box.area()));
  double log_u = log(runif_mt(rng));
  std::ostringstream log_message;
  log_message << "log_likelihood  x       y" << endl;
  double max_loglike = max(log_likelihood);
  for (int i = 0; i < log_likelihood.size(); ++i) {
    log_message << std::setw(10) << log_likelihood[i] - max_loglike
                << std::setw(10) << grid_centers[i].x() << std::setw(10)
                << grid_centers[i].y() << std::endl;
  }
  if (log_u < log_MH_ratio) {
    moveTo(candidate);
    log_message << "Grid search proposal accepted " << original_center << " -> "
                << candidate << " with log likelihood " << loglike_orig
                << " -> " << loglike_cand;
  } else {
    moveTo(original_center);
    log_message << "Grid search proposal rejected " << original_center
                << " -/> " << candidate << " with log likelihood "
                << loglike_orig << " -/> " << loglike_cand;
  }
  addLogMessage(log_message.str());
}

//===========================================================================
void Object2D::proposeMove(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  BoundingBox original_bounding_box = this->boundingBox();
  Point2D original_center = center();

  BoundingBox expanded_box;
  if (runif_mt(rng) < .5) {
    // Try big moves half the time.  Move anywhere in the region.
    expanded_box = region.intersection(BoundingBox(screen));
  } else {
    // Try small 'fine-tuning' moves half the time.  Only move a pixel or two
    // in either direction.
    expanded_box =
        BoundingBox(original_center.x() - 2, original_center.x() + 2,
                    original_center.y() - 2, original_center.y() + 2);
  }

  BoundingBox candidate_region = expanded_box.intersection(region);
  Point2D candidate_center(
      runif_mt(rng, candidate_region.left(), candidate_region.right()),
      runif_mt(rng, candidate_region.bottom(), candidate_region.top()));
  moveTo(candidate_center);
  BoundingBox candidate_bounding_box = boundingBox();

  BoxUnion likelihood_region(original_bounding_box.intersection(region),
                             candidate_bounding_box.intersection(region));
  // The proposal distribution is symmmetric, and the prior over location is
  // uniform, so the only contribution to the MH ratio is the
  // integratedLogLikelihood.

  double loglike_cand = Object2D::integratedLogLikelihood(
      screen, likelihood_region, objects, background);
  moveTo(original_center);
  double loglike_original = Object2D::integratedLogLikelihood(
      screen, likelihood_region, objects, background);

  if (std::isnan(loglike_original)) {
    std::cout << "Found a NaN" << endl;
  }
  double log_MH_ratio = loglike_cand - loglike_original;
  std::ostringstream log_message;
  double log_u = runif_mt(rng);
  if (log_u < log_MH_ratio) {
    moveTo(candidate_center);
    log_message << "MH 'move' accepted: new center is " << candidate_center
                << " (likelihood " << loglike_cand << " vs " << loglike_original
                << ") on a region with area: " << likelihood_region.area();
  } else {
    log_message << "MH 'move' from " << original_center << " to "
                << candidate_center << " rejected (log likelihood "
                << loglike_cand << " vs " << loglike_original << ")"
                << " on a region with area " << likelihood_region.area();
    moveTo(original_center);
  }
  addLogMessage(log_message.str());
}

//===========================================================================
void Circle::proposeRescale(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  // The proposal distribution can go from half to double the current value.
  // The lower limit cannot be any smaller, because you wouldn't able to get
  // back in a single move, so the move would not be reversible.
  double candidate_radius = runif_mt(rng, .5 * radius_, 2 * radius_);
  if (candidate_radius < prior_->radiusPrior()->lo() ||
      candidate_radius > prior_->radiusPrior()->hi()) {
    return;
  }

  double original_radius = radius_;
  BoundingBox original_bounding_box = boundingBox();
  radius_ = candidate_radius;
  BoundingBox candidate_bounding_box = boundingBox();
  BoundingBox box = candidate_bounding_box.super_box(original_bounding_box)
                        .intersection(region);
  double loglike_cand =
      Object2D::integratedLogLikelihood(screen, box, objects, background);
  radius_ = original_radius;
  double loglike_original =
      Object2D::integratedLogLikelihood(screen, box, objects, background);

  // The proposal distribution for moving from the original to the candidate
  // is 1 / (1.5 * original_radius).

  // The proposal distribution for the reverse move is 1 / (1.5 *
  // candidate_radius).

  // The prior is uniform.  Conditional on making it this far, the prior can
  // be ignored.

  double log_MH_ratio =
      loglike_cand - loglike_original + log(original_radius / candidate_radius);

  double log_u = log(runif_mt(rng));
  std::ostringstream log_message;
  if (log_u < log_MH_ratio) {
    radius_ = candidate_radius;
    log_message << "Rescale accepted: " << original_radius << " -> "
                << candidate_radius << " with log likelihood "
                << loglike_original << " -> " << loglike_cand;

  } else {
    radius_ = original_radius;
    log_message << "Rescale rejected: " << original_radius << " -/> "
                << candidate_radius << " with log likelihood "
                << loglike_original << " -/> " << loglike_cand;
  }
  addLogMessage(log_message.str());
}

//===========================================================================
void Circle::samplePosteriorSlice(
    BOOM::RNG &rng, const Screen2D &screen, const BoundingBox &region,
    const std::vector<std::shared_ptr<Object2D>> &objects,
    const std::shared_ptr<Background2D> &background) {
  Vector params = {x_, y_, radius_};
  std::function<double(const Vector &x)> target = [&](const Vector &x) {
    if (x[0] < region.left() || x[0] > region.right() ||
        x[1] < region.bottom() || x[1] > region.top()) {
      return negative_infinity();
    }
    double ans = prior_->radiusPrior()->logp(x[2]);
    if (!std::isfinite(ans)) {
      return ans;
    }
    this->x_ = x[0];
    this->y_ = x[1];
    this->radius_ = x[2];
    ans +=
        Object2D::integratedLogLikelihood(screen, region, objects, background);
    if (!std::isfinite(ans)) {
      std::cout << "Infinity found in Circle::samplePosterior." << std::endl;
    }
    return ans;
  };

  UnivariateSliceSampler sampler(target, 1.0, true, &rng);
  sampler.set_limits({region.left(), region.bottom(), 0},
                     {region.right(), region.top(), infinity()});

  params = sampler.draw(params);
  x_ = params[0];
  y_ = params[1];
  radius_ = params[2];
  drawColors(rng, screen);
}

void Circle::setRadius(double radius) {
  if (radius <= 0) {
    report_error("Radius must be positive.");
  }
  radius_ = radius;
}

void Circle::scaleToBoundingBoxVolume(double volume) {
  double current_volume = square(2 * radius_);
  double ratio = volume / current_volume;
  radius_ *= sqrt(ratio);
}

//===========================================================================
CirclePrior::CirclePrior(double intensity, const Ptr<UniformModel> &radius,
                         const Ptr<GaussianModelGivenSigma> &colorMean,
                         const Ptr<ChisqModel> &color_precision)
    : intensity_(intensity), radiusPrior_(radius), colorMeanPrior_(colorMean),
      colorPrecisionPrior_(color_precision) {
  if (intensity_ <= 0) {
    report_error("Intensity must be positive.");
  }
}

std::shared_ptr<Circle>
CirclePrior::simulateCircle(RNG &rng, const BoundingBox &region) const {
  double x = runif_mt(rng, region.left(), region.right());
  double y = runif_mt(rng, region.bottom(), region.top());
  double radius = radiusPrior_->sim(rng);
  return std::shared_ptr<Circle>(
      new Circle(x, y, radius, const_cast<CirclePrior *>(this)));
}

double CirclePrior::logpri(const Object2D &object) const {
  if (object.type() != ObjectType::CIRCLE) {
    report_error("Not a circle.");
  }
  const Circle &circle(static_cast<const Circle &>(object));
  return radiusPrior_->logp(circle.radius());
}

} // namespace slam
