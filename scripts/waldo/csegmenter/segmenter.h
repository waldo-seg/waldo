// segment.h


// CopyRight  2018  Daniel Povey
//                  Hang Lyu
// Apache 2.0
#include <iostream>
#include <iomanip>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <list>
#include <unordered_map>
#include <vector>
#include <queue>
#include <algorithm>
#include <functional>
#include <limits>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


using namespace std;

// This is just a simple interface to read NumpyArray
template<typename Real>
class Matrix {
 public:
  Matrix(Real* data, size_t r, size_t c): data_(data), rows_(r), cols_(c) {}
  Matrix() :data_(NULL), rows_(0), cols_(0) {}

  ~Matrix() {}

  Matrix(const Matrix<Real>& other) {
    data_ = other.data_;
    rows_ = other.rows_;
    cols_ = other.cols_;
  }

  inline Real& operator() (size_t r, size_t c) const {
    return *(data_ + r * cols_ + c);
  }

 private:
  Real* data_;
  size_t rows_, cols_;
};


// declare
class Object;
class AdjacencyRecord;
class ObjectSegmenter;


struct IntPairHasher {
  size_t operator() (const pair<int, int> a) const noexcept {
    return (size_t(a.first) * 1619 + size_t(a.second) * 4111);
  }
};


struct IntPairEqual {
  bool operator() (const pair<int, int> a,
                   const pair<int, int> b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};


typedef unordered_set<pair<int, int>, IntPairHasher, IntPairEqual> PixelSet;
/*
This class represents an "object" in the output image.
Attributes:
    object_class:  A record of the current assigned class (an integer)
    pixels:  A set of pixels (2-tuples) that are part of the object
    class_log_probs:  An array indexed by class, of the total (over all
                      pixels in this object) of the log-prob of assigning this
                      pixel to this class; the object_class corresponds to the
                      index of the max element of this.
    adjacency_list:  A list of adjacency records, recording other objects to
                     which this object is adjacent.  ('Adjacent' means "linked
                     by an offset", not adjacency in the normal sense). It's
                     actually a map (from obj pairs to adjacency record) for
                     faster search and access.
*/
class Object {
 public:
  Object(PixelSet& pixels, size_t id, ObjectSegmenter* segmenter);

  inline PixelSet* GetPixels() { return &pixels_; }

  inline int GetObjectClass() const { return object_class_; }

  inline void SetObjectClass(int obj_class) { this->object_class_ = obj_class; }
  
  inline vector<float>* GetClassLogprobs() { return &class_logprobs_; }
  
  inline void AddClassLogprobs(const vector<float>* other) {
    if(!(this->class_logprobs_.size() == other->size())) {
      cout << "Error: the logprobs of two objects isn't same dimension."
           << endl;
      exit(1);
    }
    vector<float>::iterator it_this = class_logprobs_.begin();
    vector<float>::const_iterator it_other = other->begin();
    for (; it_this != class_logprobs_.end(); it_this++, it_other++) {
      *it_this += *it_other;
    }
  }

  inline float GetClassLogprob() const { 
    return class_logprobs_[object_class_];
  }

  inline size_t GetId() const { return id_; }

  inline float GetSamenessLogprob() const { return sameness_logprob_; }
  inline void AddSamenessLogprob(float other) {
    this->sameness_logprob_ += other;
  }

  inline unordered_map<size_t, AdjacencyRecord*>* GetAdjacencyList() {
    return &adjacency_list_;
  }

  inline bool operator== (const Object& other) {
    return (id_ == other.id_);
  }

  inline bool operator!= (const Object& other) {
    return (id_ != other.id_);
  }

 private:
  PixelSet pixels_;
  size_t id_;
  int object_class_;
  vector<float> class_logprobs_;
  float sameness_logprob_ = 0.0;
  unordered_map<size_t, AdjacencyRecord*> adjacency_list_;
};


/*
This class implements an adjacency record with functions for computing object
merge log-probs and class merge log-probs.

Attributes:
  obj1, obj2:  The two objects to which it refers
  object_merge_log_prob:  This is the change in log-probability from merging
               these two objects, without considering the effect arising from
               changes of class assignments. This is the sum of the following:
               For each p,o such that o is in "offsets", p is in one of the two
               objects and p+o is in the other, the value
               log(p(same) / p(different)), i.e.g log(b_{p,o} / (1-b{p,o})).
               Note: if the sum above had no terms in it, this adjacency record
               should not exist because the objects would not be "adjacent" in
               the sense which we refer to.
  merge_priority:  This merge priority is a heuristic which will determine what
               kinds of objects will get merged first, and is a key choice that
               we'll have to experiment with.  (Note: you can change the sign
               if it turns out to be easier for python heap reasons).  The
               general idea will be: merge_priority = merge_log_prob / den
               where merge_log_prob is the log-prob change from doing this
               merge, and for example, "den" might be the maximum of the
               num-pixels in object1 and object2.  We can experiment with
               different heuristics for "den" though.
  class_delta_logprob:  It is a term representing a change in the total log-prob
               that we'll get from merging the two objects, that arises from
               forcing the class assignments to be the same.  If the two objects
               already have the same assigned class, this will be zero.  If
               different, then this is a value <= 0 which can be obtained by
               summing the objects' 'class_logprobs' arrays, finding the largest
               log-prob in the total, and subtracting the total from the current
               class-assignments of the two objects.
  merged_class:  The class that the merged object would have, obtained when
               figuring out class_delta_log_prob.
*/
class AdjacencyRecord {
 public:
  AdjacencyRecord(Object* obj1, Object* obj2, ObjectSegmenter* segmenter,
                  pair<int, int>* pixel, pair<int, int>* offset);

  void ComputeObjMergeLogprob(ObjectSegmenter* segmenter);

  void ComputeClassDeltaLogprob();

  void UpdateMergePriority(ObjectSegmenter* segmenter);

  size_t GetHashValue() const { return cached_hash_; }

  inline float GetPriority() const { return merge_priority_; }

  inline void SetPriority(float value) {
    this->merge_priority_ = value;
  }

  inline float GetDifferentnessLogprob() const {
    return differentness_logprob_;
  }

  inline void AddDifferentnessLogprob(float other) {
    this->differentness_logprob_ += other;
  }

  inline float GetSamenessLogprob() const { return sameness_logprob_; }

  inline void AddSamenessLogprob(float other) {
    this->sameness_logprob_ += other;
  }

  inline float GetObjMergeLogprob() const { return obj_merge_logprob_; }

  inline void AddObjMergeLogprob(float other) {
    this->obj_merge_logprob_ += other;
  }

  inline int GetMergedClass() const { return merged_class_; }

  inline Object* GetObj1() const { return obj1_; }

  inline void SetObj1(Object* other) { this->obj1_ = other; }

  inline Object* GetObj2() const { return obj2_; }

  inline void SetObj2(Object* other) { this->obj2_ = other; }

  inline float GetClassDeltaLogprob() const { return class_delta_logprob_; }
 
  void SortAndUpdateHash();
 private:
  Object* obj1_;
  Object* obj2_;
  size_t cached_hash_;

  float differentness_logprob_;
  float sameness_logprob_;
  float obj_merge_logprob_ = numeric_limits<float>::min();

  float class_delta_logprob_;
  int merged_class_;

  float merge_priority_;
};


// Hash function for AdjacencyRecord. It converts AdjacencyRecord to hash
// code by looking at the id of two Objects.
struct AdjacencyRecordHasher {
  size_t operator()(AdjacencyRecord* arec) {
    return (arec->GetObj1()->GetId() * 1619 +
            arec->GetObj2()->GetId() * 3203);
  }
};


struct ObjectSegmenterOption {
  float same_different_bias;
  float object_merge_factor;
  float merge_logprob_bias;

  ObjectSegmenterOption():
    same_different_bias(0.0),
    object_merge_factor(1.0),
    merge_logprob_bias(0.0) {
  }

  ObjectSegmenterOption(float same_different_bias,
                        float object_merge_factor,
                        float merge_logprob_bias):
    same_different_bias(same_different_bias),
    object_merge_factor(object_merge_factor),
    merge_logprob_bias(merge_logprob_bias) {}

  ObjectSegmenterOption(ObjectSegmenterOption& opt) {
    same_different_bias = opt.same_different_bias;
    object_merge_factor = opt.object_merge_factor;
    merge_logprob_bias = opt.merge_logprob_bias;
  }
};

struct PriorityCompare {
  bool operator() (const pair<float, AdjacencyRecord*>& a,
                   const pair<float, AdjacencyRecord*>& b) {
    return (a.first < b.first);
  }
};


class ObjectSegmenter {
 public:
  ObjectSegmenter(float* nnet_class_pred, int class_dim,
                  float* nnet_sameness_probs, int offset_dim,
                  int img_width, int img_height, int num_classes,
                  int* offset_list,
                  int* output,
                  int* object_class,
                  const ObjectSegmenterOption& opts,
                  int verbose = 0);

  ~ObjectSegmenter();

  void InitObjectsAndAdjacencyRecords();

  inline int GetNumClasses() const { return num_classes_; }

  inline float GetClassLogprob(const pair<int,int>& pixel,
                               int class_index) {
    return log((class_probs_.at(class_index))(pixel.first, pixel.second));
  }

  inline vector<pair<int, int> >* GetOffsets() { return &offsets_; }

  inline float GetSamenessProb(pair<int,int> pixel,
                               pair<int, int> offset) const {
    return (sameness_probs_.at(offset))(pixel.first, pixel.second);
  }

  inline const ObjectSegmenterOption* GetSegmenterOption() const {
    return &opts_;
  }

  void ShowStats();

  float ComputeTotalLogprob();

  void Visualize();

  bool Debug();

  void RunSegmentation();

  void OutputMask();

  void Merge(AdjacencyRecord* arec);

  void ComputeTotalLogprobFromScratch();

 private:
  unordered_map<int, Matrix<float> > class_probs_;
  unordered_map<pair<int,int>, Matrix<float>, IntPairHasher,
    IntPairEqual> sameness_probs_;
  vector<pair<int, int> > offsets_;
  int class_dim_, offset_dim_, img_width_, img_height_, num_classes_;
  unordered_map<size_t, Object*> objects_;
  unordered_map<pair<int, int>, Object*, IntPairHasher,
    IntPairEqual> pixel2obj_;
  unordered_map<size_t, AdjacencyRecord*> adjacency_records_;
  priority_queue<pair<float, AdjacencyRecord*>, 
    vector<pair<float, AdjacencyRecord*> >, PriorityCompare> segmenter_queue_;

  Matrix<int> output_;
  Matrix<int> object_class_;

  const ObjectSegmenterOption& opts_;
  int verbose_;
};
