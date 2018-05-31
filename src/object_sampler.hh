#include "exp/user/stb/slam/objects.hh"
#include "exp/user/stb/slam/screen2d.hh"

#include "Samplers/MoveAccounting.hpp"

namespace slam {

// An ObjectSampler manages MCMC updates for collections of objects.
class ObjectSampler {
public:
  ObjectSampler(const std::shared_ptr<Background2D> &background,
                int gridSearchSize = 30);

  // Add an object type to the dictionary of available object types.
  void addObjectType(const std::shared_ptr<ObjectPrior2D> &objectPrior);

  // Return the prior distribution associated with a given object type.
  //
  // Args:
  //   type:  The type of object for which the prior is desired.
  //
  // Returns:
  //   If a prior distribution for the requested object type has previously
  //   been added then it is returned.  Otherwise nullptr is returned.
  std::shared_ptr<ObjectPrior2D> getPrior(ObjectType type) const;

  // Scan a region of previously unobserved pixels for new objects.  The scan
  // is implemented using an MCMC algorithm for a prescribed number of
  // iterations.
  //
  // Args:
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   niter: The desired number of MCMC iterations.
  //
  // Effects:
  //   The data member objectRecord_ is cleared, and repopulated by the
  //   results of the MCMC run.
  void scanForNewObjects(BOOM::RNG &rng, const Screen2D &screen,
                         const BoundingBox &region, int niter);

  // Run an MCMC update on each object, allowing it to change its location and
  // size.
  //
  // Args:
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   objects: The collection of objects to move.
  void moveAndResizeObjects(BOOM::RNG &rng, const Screen2D &screen,
                            const BoundingBox &region,
                            std::vector<std::shared_ptr<Object2D>> &objects);

  // Choose a random object from the set of objects and propose changing its
  // type to another object.
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   objects: The collection of objects.  One of these might change type.
  void morphObjects(BOOM::RNG &rng, const Screen2D &screen,
                    const BoundingBox &region,
                    std::vector<std::shared_ptr<Object2D>> &objects);

  // Attempt to morph the object at 'index' in 'objects'.
  void morphSingleObject(int index, BOOM::RNG &rng, const Screen2D &screen,
                         const BoundingBox &region,
                         std::vector<std::shared_ptr<Object2D>> &objects);

  // Choose a random object and either attempt to split it into two objects,
  // or combine it with an object with which it overlaps.
  //
  // Args:
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   objects: The collection of objects.  One of these might change type.
  void splitCombineObjects(BOOM::RNG &rng, const Screen2D &screen,
                           const BoundingBox &region,
                           std::vector<std::shared_ptr<Object2D>> &objects);

  // Implementation for splitCombineObjects.
  void splitMove(BOOM::RNG &rng, const Screen2D &screen,
                 const BoundingBox &region,
                 std::vector<std::shared_ptr<Object2D>> &objects);
  void combineMove(BOOM::RNG &rng, const Screen2D &screen,
                   const BoundingBox &region,
                   std::vector<std::shared_ptr<Object2D>> &objects);

  //
  // Args:
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   objects: The collection of objects.  One of these might change type.
  void addRemoveObjects(BOOM::RNG &rng, const Screen2D &screen,
                        const BoundingBox &region,
                        std::vector<std::shared_ptr<Object2D>> &objects);

  // Implementation for the 'add' part of addRemoveObjects.
  void addObjectMove(BOOM::RNG &rng, const Screen2D &screen,
                     const BoundingBox &region,
                     std::vector<std::shared_ptr<Object2D>> &objects);

  // Implementation for the 'remove' part of addRemoveObjects.
  void removeObjectMove(BOOM::RNG &rng, const Screen2D &screen,
                        const BoundingBox &region,
                        std::vector<std::shared_ptr<Object2D>> &objects);

  // Create a random object from among the set of object types.
  // Args:
  //   rng:  The random number generator.
  //   currentObject:  The object to be morphed.
  //
  // Returns:
  //   A newly allocated object, of a different type than currentObject.  The
  //   new object is located in the same position as currentObject.  The
  //   remaining parameters of the new object are sampled from the prior.
  std::shared_ptr<Object2D>
  randomMorph(BOOM::RNG &rng, const std::shared_ptr<Object2D> &currentObject);

  // Create a random object of random type, with values simulated from its
  // prior.
  //
  // Args:
  //   rng:  The random number generator.
  //   region:  The region in which the object is to be placed.
  // Returns:
  //   An object randomly generated from its prior.  The object type is chosen
  //   from a multinomial distribution with probabilities proportional to the
  //   Poisson intensity associated with each object type.
  std::shared_ptr<Object2D> newRandomObject(BOOM::RNG &rng,
                                            const BoundingBox &region);

  // Returns the sum of the Poisson intensity parameters from the priors for
  // all available object types.
  double overallIntensity() const;

  void clearObjectRecord() { objectRecord_.clear(); }

private:
  void recordObjects(const std::vector<std::shared_ptr<Object2D>> &objects);

  // Fill the 'objects' with a collection of randomly generated objects.  The
  // search region is divided into squares into squares of side length
  // gridSearchSize_, and one randomly chosen object is created in each
  // square.  It is expected that most objects will be quickly killed off by
  // subsequent MCMC steps, and that surviving objects will quickly morph into
  // the right types.
  //
  // Args:
  //   rng:  The random number generator.
  //   screen: The screen containing the image being analyzed.  A subset of
  //     the screen contains previously unobserved pixels.
  //   region: The region of the screen containing previously unobserved
  //     pixels.
  //   objects: On entry this is expected to be empty.  New objects, randomly
  //     generated from the prior, will be added.  One per grid square.  On
  //     exit this argument contains the new objects.
  std::vector<std::shared_ptr<Object2D>>
  initializeObjectSearch(BOOM::RNG &rng, const Screen2D &screen,
                         const BoundingBox &region);

  std::shared_ptr<Background2D> background_;
  int gridSearchSize_;

  std::vector<std::shared_ptr<ObjectPrior2D>> objectPriors_;
  std::map<ObjectType, int> priorLocations_;

  std::vector<std::vector<std::shared_ptr<Object2D>>> objectRecord_;

  BOOM::MoveAccounting mhMoveResults_;
};

} // namespace slam
