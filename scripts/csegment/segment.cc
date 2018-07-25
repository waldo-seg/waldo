// segment.cc

#include "segment.h"

Object::Object(PixelSet& pixels, size_t id, 
               ObjectSegmenter* segmenter):
  pixels(pixels), id(id) {
  int num_classes = segmenter->GetNumClasses();
  class_logprobs.reserve(num_classes);
  for (int i = 0; i < num_classes; i++) {
    float cur_prob = 0.0;
    for (PixelSet::iterator iter = pixels.begin();
         iter != pixels.end(); iter++) {
      cur_prob += segmenter->GetClassLogprob(*iter, i);
    }
    class_logprobs.push_back(cur_prob);
  }
  object_class = distance(class_logprobs.begin(),
                          max_element(class_logprobs.begin(),
                                      class_logprobs.end()));
}


AdjacencyRecord::AdjacencyRecord(Object* obj1, Object* obj2,
                                 ObjectSegmenter* segmenter,
                                 pair<int, int>* pixel,
                                 pair<int, int>* offset) :
  obj1(obj1), obj2(obj2) {

  SortAndUpdateHash();
  // Without iterating the whole offsets, it will save a little time.
  if (pixel != NULL && offset != NULL) {
    float same_prob = segmenter->GetSamenessProb(*pixel, *offset);
    differentness_logprob = log(1.0 - same_prob);
    sameness_logprob = log(same_prob);
    obj_merge_logprob = sameness_logprob - differentness_logprob;
  } else {
    ComputeObjMergeLogprob(segmenter);
    if (obj_merge_logprob == numeric_limits<float>::min()) {
      cout << "Error: Bad adjacency record. The given objects are not adjacent."
           << endl;
      exit(1);
    } 
  }
  UpdateMergePriority(segmenter);
}


void AdjacencyRecord::SortAndUpdateHash() {
  if (obj1->GetId() > obj2->GetId()) {
    Object* tmp = obj1;
    obj1 = obj2;
    obj2 = tmp;
  }
  cached_hash = AdjacencyRecordHasher()(this);
}


void AdjacencyRecord::ComputeObjMergeLogprob(ObjectSegmenter* segmenter) {
  float logprob = 0.0;
  bool has_adjacency = false;
  vector<pair<int, int> >::iterator offset_iter = 
                                    segmenter->GetOffsets()->begin(),
                                    offset_end =
                                    segmenter->GetOffsets()->end();

  PixelSet::iterator obj1_iter = obj1->GetPixels()->begin(),
                     obj1_end = obj1->GetPixels()->end(),
                     obj2_iter = obj2->GetPixels()->begin(),
                     obj2_end = obj2->GetPixels()->end();
  for (; offset_iter != offset_end; offset_iter++) {
    for (; obj1_iter != obj1_end; obj1_iter++) {
      pair<int, int> p1(obj1_iter->first, obj1_iter->second);
      pair<int, int> p2(p1.first + offset_iter->first,
                        p1.second + offset_iter->second);
      if (obj2->GetPixels()->find(p2) != obj2_end) {
        has_adjacency = true;
        float same_prob = segmenter->GetSamenessProb(p1, *offset_iter);
        float log_same_prob = log(same_prob);
        float log_different_prob = log(1.0 - same_prob);
        sameness_logprob += log_same_prob;
        differentness_logprob += log_different_prob;
        logprob += (log_same_prob - log_different_prob); 
      }
    }
    for (; obj2_iter != obj2_end; obj2_iter++) {
      pair<int, int> p1(obj2_iter->first, obj2_iter->second);
      pair<int, int> p2(p1.first + offset_iter->first,
                        p1.second + offset_iter->second);
      if (obj1->GetPixels()->find(p2) != obj1_end) {
        has_adjacency = true;
        float same_prob = segmenter->GetSamenessProb(p1, *offset_iter);
        float log_same_prob = log(same_prob);
        float log_different_prob = log(1.0 - same_prob);
        sameness_logprob += log_same_prob;
        differentness_logprob += log_different_prob;
        logprob += (log_same_prob - log_different_prob); 
      }
    }
  }
  if (has_adjacency) {
    obj_merge_logprob = logprob;
  }
}


void AdjacencyRecord::ComputeClassDeltaLogprob() {
  if (obj1->GetObjectClass() == obj2->GetObjectClass()) {
    class_delta_logprob = 0.0;
    merged_class = obj1->GetObjectClass();
  } else {
    size_t num_classes = obj1->GetClassLogprobs()->size();
    if(num_classes != obj2->GetClassLogprobs()->size()) {
      cout << "obj1 id is: obj" << obj1->GetId()
           << " and obj2 id is: obj" << obj2->GetId() << endl;
      cout << "the size of obj1 is " << obj1->GetClassLogprobs()->size()
           << " and the size of obj2 is "  
           << obj2->GetClassLogprobs()->size() << endl;
    }
    vector<float> joint_class_logprobs;
    for (size_t i = 0; i < num_classes; i++) {
      joint_class_logprobs.push_back((*(obj1->GetClassLogprobs()))[i] +
                                     (*(obj2->GetClassLogprobs()))[i]);
    }
    merged_class = distance(joint_class_logprobs.begin(),
                            max_element(joint_class_logprobs.begin(),
                                        joint_class_logprobs.end()));
    class_delta_logprob = joint_class_logprobs[merged_class] -
                          obj1->GetClassLogprob() - obj2->GetClassLogprob();
  }
}


void AdjacencyRecord::UpdateMergePriority(ObjectSegmenter* segmenter) {
  ComputeClassDeltaLogprob();
  size_t den = obj1->GetPixels()->size() * obj2->GetPixels()->size();
  merge_priority =  (obj_merge_logprob *
                     segmenter->GetSegmenterOption()->object_merge_factor +
                     class_delta_logprob +
                     segmenter->GetSegmenterOption()->merge_logprob_bias) /
                     den;
}


ObjectSegmenter::ObjectSegmenter(float* nnet_class_pred, int class_dim,
                                 float* nnet_sameness_probs, int offset_dim,
                                 int img_width, int img_height, int num_classes,
                                 int* offset_list, int* output,
                                 int* object_class,
                                 const ObjectSegmenterOption& opts,
                                 int verbose):
  class_dim(class_dim), offset_dim(offset_dim), img_width(img_width),
  img_height(img_height), num_classes(num_classes), opts(opts), verbose(verbose)
{
  this->output = Matrix<int>(output, img_height, img_width);
  this->object_class = Matrix<int>(object_class, 1, img_height * img_width);
  // Initialize offsets
  for (int i = 0; i < offset_dim; i++) {
    offsets.push_back(make_pair(
          *(offset_list + i * 2), *(offset_list + i*2 + 1)));
  }
  // Initialize class_probs
  class_probs.reserve(class_dim);
  for (int i = 0; i < class_dim; i++) {
    class_probs[i] = Matrix<float>(nnet_class_pred + (img_width*img_height) * i,
                                   img_height, img_width);
  }
  // Initialize sameness_probs
  sameness_probs.reserve(offset_dim);
  for (int i = 0; i < offset_dim; i++) {
    sameness_probs[offsets[i]] = Matrix<float>(
        nnet_sameness_probs + (img_width*img_height) * i,
        img_height, img_width);
  }
  if (opts.same_different_bias != 0) {
    for (int i = 0; i < offset_dim; i++) {
      for (int row = 0; row < img_height; row++) {
        for (int col = 0; col < img_width; col++) {
          float* cur_sameness_probs = &(sameness_probs[offsets[i]](row, col));
          float sameness_probs_biased_logit =
            log(*cur_sameness_probs) - log(1.0 - *cur_sameness_probs) +
            opts.same_different_bias;
          *cur_sameness_probs = 1.0 / (1.0 + exp(-sameness_probs_biased_logit));
        }
      }
    }
  }
  // Initialize pixel2obj and objects
  size_t obj_id = 0;
  for (int row = 0; row < img_height; row++) {
    for (int col = 0; col < img_width; col++) {
      PixelSet pixel;
      pixel.insert(make_pair(row, col));
      Object* obj = new Object(pixel, obj_id, this);
      objects[obj_id] = obj;
      pixel2obj[make_pair(row, col)] = obj;
      obj_id++;
    }
  }
  // Initialize adjacency_records
  for (int row = 0; row < img_height; row++) {
    for (int col = 0; col < img_width; col++) {
      pair<int, int> pixel = make_pair(row, col);
      Object *obj1 = pixel2obj[pixel];
      for (vector<pair<int,int> >::iterator iter = offsets.begin();
           iter != offsets.end(); iter++) {
        if (0 <= (row + iter->first) && (row + iter->first) < img_height &&
            0 <= (col + iter->second) && (col + iter->second) < img_width) {
          Object *obj2 = pixel2obj[make_pair(row + iter->first,
                                             col + iter->second)];
          AdjacencyRecord *arec = new AdjacencyRecord(obj1, obj2, this,
              &pixel, &(*iter));
          adjacency_space.push_back(arec);
          adjacency_records[arec->GetHashValue()] = arec;
          (*(obj1->GetAdjacencyList()))[arec->GetHashValue()] = arec;
          (*(obj2->GetAdjacencyList()))[arec->GetHashValue()] = arec;
          if (arec->GetPriority() >= 0) {
            segmenter_queue.push(make_pair(arec->GetPriority(), arec));
          }
        }
      }
    }
  }
}



void ObjectSegmenter::ShowStats() {
  cout << "Total logprob (not incl. sameness inside objects): "
       << ComputeTotalLogprob() << endl;
//       << setprecision(3) << ComputeTotalLogprob() << endl;

  cout << "Total number of objects: " << objects.size() << endl;
  cout << "Total number of adjacency records: "
       << adjacency_records.size() << endl;
  cout << "Total number of records in the queue: "
       << segmenter_queue.size() << endl;
  vector<int> object_length;
  vector<int> adjacency_length;
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    object_length.push_back(iter->second->GetPixels()->size());
    adjacency_length.push_back(iter->second->GetAdjacencyList()->size());
  }
  sort(object_length.begin(), object_length.end());
  reverse(object_length.begin(), object_length.end());
  sort(adjacency_length.begin(), adjacency_length.end());
  reverse(adjacency_length.begin(), adjacency_length.end());
  int len = min(object_length.size(), (size_t)10);

  cout << "Top 10 biggest objs (#pixels): ";
  for (int i = 0; i < len; i++) {
    cout << object_length[i] << " ";
  }
  cout << endl;
  cout << "Top 10 biggest objs (adj_list size): ";
  for (int i = 0; i < len; i++) {
    cout << adjacency_length[i] << " ";
  }
  cout << endl;
}


float ObjectSegmenter::ComputeTotalLogprob() {
  float tot_class_logprob = 0.0;
  float tot_differentness_logprob = 0.0;
  float tot_sameness_logprob = 0.0;
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    tot_class_logprob += iter->second->GetClassLogprob();
    tot_sameness_logprob += iter->second->GetSamenessLogprob();
  }
  for (unordered_map<size_t, AdjacencyRecord*>::iterator iter =
       adjacency_records.begin(); iter != adjacency_records.end(); iter++) {
    tot_differentness_logprob += iter->second->GetDifferentnessLogprob();
  }
  return tot_class_logprob + (tot_differentness_logprob +
                              tot_sameness_logprob) * opts.object_merge_factor;
}


void ObjectSegmenter::Visualize() {
  for (int i = 0; i < img_height; i++) {
    for (int j = 0; j < img_width; j++) {
      output(i,j) = 0;
    }
  }
  int k = 1;
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    PixelSet::iterator pix_iter = iter->second->GetPixels()->begin(),
                                  pix_end = iter->second->GetPixels()->end();
    int tot_row = 0, tot_col = 0, count = 0;
    for (; pix_iter != pix_end; pix_iter++) {
      output(pix_iter->first, pix_iter->second) = k;
      tot_row += pix_iter->first;
      tot_col += pix_iter->second;
      count++;
    }
    output(int(tot_row/count),int(tot_col/count)) = 0;
    k++;
  }
}


void ObjectSegmenter::ComputeTotalLogprobFromScratch() {
  float tot_class_logprob = 0.0;
  float tot_differentness_logprob = 0.0;
  float tot_sameness_logprob = 0.0;
  for (unordered_map<size_t, Object*>::iterator it = objects.begin();
       it != objects.end(); it++) {
    Object* obj = it->second;
    for (PixelSet::iterator pix_it = obj->GetPixels()->begin();
         pix_it != obj->GetPixels()->end(); pix_it++) {
      pair<int, int> p = *pix_it;
      pixel2obj[p] = obj;
      tot_class_logprob += GetClassLogprob(p, obj->GetObjectClass());
    }
  }
  for (int row = 0; row < img_height; row++) {
    for (int col = 0; col < img_width; col++) {
      pair<int, int> p1(row, col);
      Object* obj1 = pixel2obj[p1];
      for (vector<pair<int,int> >::iterator iter = offsets.begin();
           iter != offsets.end(); iter++) {
        if (0 <= (row + iter->first) && (row + iter->first) < img_height &&
            0 <= (col + iter->second) && (col + iter->second) < img_width) {
          Object *obj2 = pixel2obj[make_pair(row + iter->first,
                                             col + iter->second)];
          if (obj1->GetId() == obj2->GetId()) {
            tot_sameness_logprob += log(GetSamenessProb(p1, *iter));
          } else {
            tot_differentness_logprob += log(1.0 - GetSamenessProb(p1, *iter));
          }
        }
      }
    }
  }
  cout << "Final logprob from scratch: "
       << tot_class_logprob + (tot_differentness_logprob +
                              tot_sameness_logprob) * opts.object_merge_factor;
}


/*
  Do some sanity checks and make sure certain quantities have values that
  they should have.
  This function is quite time-consuming and should not be called too many times.
*/
bool ObjectSegmenter::Debug() {
  // Use output matrix as a temporary storage space
  for (int i = 0; i < img_height; i++) {
    for (int j = 0; j < img_width; j++) {
      output(i,j) = 0;
    }
  }
  // Check if the current set of objects exactly cover the whole image
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    PixelSet::iterator pix_iter = iter->second->GetPixels()->begin(),
                       pix_end = iter->second->GetPixels()->end();
    for (; pix_iter != pix_end; pix_iter++) {
      output(pix_iter->first, pix_iter->second) = 1;
    }
  }
  for (int i = 0; i < img_width; i++) {
    for (int j = 0; j < img_height; j++) {
      if (output(i,j) != 1) {
        cout << "Error: pixels are not all covered or they are double counted."
             << endl;
        return false;
      }
    }
  }
  // Check the adjacency lists of the objects
  int tot_obj_adj_records = 0;
  unordered_map<size_t, AdjacencyRecord*>::iterator it;

  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    tot_obj_adj_records += iter->second->GetAdjacencyList()->size();
    unordered_map<size_t, AdjacencyRecord*>::iterator arec_iter =
             iter->second->GetAdjacencyList()->begin(), arec_end =
             iter->second->GetAdjacencyList()->end();
    for(; arec_iter != arec_end; arec_iter++) {
      it = find(adjacency_records.begin(), adjacency_records.end(), *arec_iter);
      if (it == adjacency_records.end()) { return false; }

      if (*(iter->second) != *(arec_iter->second->GetObj1()) &&
          *(iter->second) != *(arec_iter->second->GetObj2())) { return false; }

      // make shure that re-computing obj-merge-logprob does not change it.
      // this is too costly to run, so only do it randomly with a small chance.
      if ((rand()%100) > 95) {
        float obj_merge_logprob = arec_iter->second->GetObjMergeLogprob();
        arec_iter->second->ComputeObjMergeLogprob(this);
        float obj_merge_logprob_new = arec_iter->second->GetObjMergeLogprob();
        if (obj_merge_logprob - obj_merge_logprob_new > 0.001) {
          cout << "Error: re-computing obj-merge logprob changed it for arec."
               << " Old logprob is " << obj_merge_logprob
               << " New logprob is " << obj_merge_logprob_new
               << endl;
          return false;
        }
      }
    }
  }
  if ((size_t)tot_obj_adj_records != (adjacency_records.size() * 2)) { 
    return false;
  }
  return true;
}


void ObjectSegmenter::OutputMask() {
  for (int i = 0; i < img_height; i++) {
    for (int j = 0; j < img_width; j++) {
      output(i,j) = 0;
    }
  }
  int pic_size = img_height * img_width;
  for (int i = 0; i < pic_size; i++) {
    object_class(0, i) = -1;
  }

  int k = 1;
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    // skip background object
    if (iter->second->GetObjectClass() == 0) {
      continue;
    }
    object_class(0, k-1) = iter->second->GetObjectClass();
    PixelSet::iterator pix_iter = iter->second->GetPixels()->begin(),
                       pix_end = iter->second->GetPixels()->end();
    for (; pix_iter != pix_end; pix_iter++) {
      output(pix_iter->first, pix_iter->second) = k;   
    }
    k++;
  }
}


/*
This is the top-level function that performs the optimization.
This is the overview:
   - While the queue is non-empty:
     - Pop (merge_priority, arec) from the queue.
     - If merge_priority != arec.merge_priority
       continue   # don't worry, the queue will have the right
                  # merge_priority for this arec somewhere else in it.
     - Recompute arec.merge_priority, which involves recomputing
       class_delta_log_prob.  This is needed because as we
       merge objects, the value of class_delta_log_prob and/or the
       number of pixels may have changed and the adjacency record
       may not have been updated.
     - If the newly computed arec.merge_priority is >= the old value (i.e. this
       merge is at least as good a merge as we thought it was when
       we got it from the queue), go ahead and merge the objects.
     - Otherwise if arec.merge_priority >=0 then re-insert "arec" into the
       queue with its newly computed merge priority.
*/
void ObjectSegmenter::RunSegmentation() {
  cout << "Starting segmentation..." << endl;
  int n = 0;
  while (!segmenter_queue.empty()) {
    if (verbose >= 0) {
      if (n % 500000 == 0) {
        cout << "At iteration: " << n << endl;
        ShowStats();
      }
    }
    n += 1;

    float merge_priority = segmenter_queue.top().first;
    AdjacencyRecord* arec = segmenter_queue.top().second;
    segmenter_queue.pop();
    if (merge_priority != arec->GetPriority()) {
      continue;
    }
    if (arec->GetObj1() == NULL || arec->GetObj2() == NULL) {
      continue;
    } 
    arec->UpdateMergePriority(this);
    if (arec->GetPriority() >= merge_priority) {
      Merge(arec);
    } else if (arec->GetPriority() >= 0) {
      segmenter_queue.push(make_pair(arec->GetPriority(), arec));
    }
  }
  cout << "Finished. Queue is empty." << endl;
  ShowStats();
  // ComputeTotalLogprobFromScratch();
  // Visualize();
  OutputMask();
}


/*
This is the most nontrivial aspect of the algorithm: how to merge objects.
The basic steps in this function are as follows:
  - Swap object1 and object2 as necessary to ensure that the
    num-pixels in object1 is >= the num-pixels in object2.  We will
    assimilate object2 into object1 and we can let object2 be deleted.
  - Set object1's object_class to merged_class
  - Append object2's pixels to object1's pixels
  - Add object2's class_log_probs to object1's class_log_probs
  - Merge object2's adjancency records into object1's adjacency
    records: more specifically,
  - For each element "this_arec" in object2.adjacency_list, change
    whichever of this_arec.object1 or this_arec.object2 equals "object2"
    to "object1".  That is, make it point to the merged object
    "object1", instead of to the doomed "object2".
  - If object1.adjacency_list already contains an adjacency record with
    same pair of objects that are now in this_arec (viewing them as an
    unordered pair), then add this_arec.object_merge_log_prob to that
    adjacency record's object_merge_log_prob.  Otherwise, add this_arec
    to object1.adjacency_list.
  - For each adjacency record that is directly touched during the
    process above:
    - Recompute its class_delta_log_prob, merged_class and
      merge_priority; if its merge_priority has changed and is >= 0,
      re-insert it into the queue.
*/
void ObjectSegmenter::Merge(AdjacencyRecord* arec) {
  bool do_debugging = false;
  Object *obj1 = arec->GetObj1();
  Object *obj2 = arec->GetObj2();
  if (obj1 == NULL || obj2 == NULL) {
    return;
  }
  if (*obj1 == *obj2) {
    return;
  }
  if (obj1->GetPixels()->size() < obj2->GetPixels()->size()) {
    Object *temp = obj1;
    obj1 = obj2;
    obj2 = temp;
  }

  //if(!(fabsf(arec->GetObjMergeLogprob() - arec->GetSamenessLogprob() +
  //           arec->GetDifferentnessLogprob()) < 0.001)) {
  //  cout << "This AdjacencyRecord shouldn't be merged" << endl;
  //  exit(1);
  //}

  if (do_debugging) {
    float old_logprob = arec->GetObjMergeLogprob();
    arec->ComputeObjMergeLogprob(this);
    if (fabsf(arec->GetObjMergeLogprob() - old_logprob) > 0.001) {
      cout << "Error: object merge logprob changed unexpectedly. "
           << arec->GetObjMergeLogprob() << " != " << old_logprob << endl;
      exit(1);
    }
  }

  // now we are sure that obj1 has equal/more pixels
  obj1->SetObjectClass(arec->GetMergedClass());
  for (PixelSet::iterator iter = obj2->GetPixels()->begin();
       iter != obj2->GetPixels()->end(); iter++) {
    obj1->GetPixels()->insert(*iter);
  }
  obj1->AddClassLogprobs(obj2->GetClassLogprobs());
  obj1->AddSamenessLogprob(arec->GetSamenessLogprob() +
                           obj2->GetSamenessLogprob());

  // remove this arec from some lists
  adjacency_records.erase(arec->GetHashValue());
  obj1->GetAdjacencyList()->erase(arec->GetHashValue());
  obj2->GetAdjacencyList()->erase(arec->GetHashValue());

  // Update the adjacencyrecords about obj2.
  for (unordered_map<size_t, AdjacencyRecord*>::iterator this_arec_it =
       obj2->GetAdjacencyList()->begin(); this_arec_it !=
       obj2->GetAdjacencyList()->end(); this_arec_it++) {
    AdjacencyRecord* this_arec = this_arec_it->second;
    // In this_arec, one is obj2.
    // Regard another one as obj3, and use obj1 instead obj2
    // But other value in "this_arec" is (obj2, obj3)
    Object *obj3 = NULL;
   
    if (*obj2 == *(this_arec->GetObj1()) ) {
      obj3 = this_arec->GetObj2();
      this_arec->SetObj1(obj1);
    } else if (*obj2 == *(this_arec->GetObj2())) {
      obj3 = this_arec->GetObj1();
      this_arec->SetObj2(obj1);
    } else {
      cout << "Error: This is not a effective Adjacency list." << endl;
      exit(1);
    }
    
    if(*obj1 == *obj3) {
      cout << "Error: cyclic merging." << endl;
      exit(1);
    }
    // The "old_hashvalue" is used to index map
    size_t old_hashvalue = this_arec->GetHashValue();

    this_arec->SortAndUpdateHash();
    
    // As obj2 is deleted, so remove this_arec from some lists
    adjacency_records.erase(old_hashvalue);
    obj3->GetAdjacencyList()->erase(old_hashvalue);
   
    // If previous has an AdjacencyRecord(obj1[without obj2], obj3),
    // update it (add this_arec value to it).
    if ( (obj1->GetAdjacencyList())->find(this_arec->GetHashValue()) !=
          obj1->GetAdjacencyList()->end() ) {
      AdjacencyRecord* that_arec =
        (*(obj1->GetAdjacencyList()))[this_arec->GetHashValue()];

      that_arec->AddObjMergeLogprob(this_arec->GetObjMergeLogprob());
      that_arec->AddDifferentnessLogprob(this_arec->GetDifferentnessLogprob());
      that_arec->AddSamenessLogprob(this_arec->GetSamenessLogprob());
      // make sure it is practically deleted from the priority queue
      this_arec->SetPriority(numeric_limits<float>::min());
      that_arec->UpdateMergePriority(this);
      if (that_arec->GetPriority() >= 0) {
         segmenter_queue.push(make_pair(that_arec->GetPriority(), that_arec));
      }
    } else {
      (*(obj1->GetAdjacencyList()))[this_arec->GetHashValue()] = this_arec;
      (*(obj3->GetAdjacencyList()))[this_arec->GetHashValue()] = this_arec;
      adjacency_records[this_arec->GetHashValue()] = this_arec;
      this_arec->UpdateMergePriority(this);
      if (this_arec->GetPriority() >= 0) {
         segmenter_queue.push(make_pair(this_arec->GetPriority(), this_arec));
      }
    }
  }
  if (verbose >= 1) {
    cout << "Deleting obj" << obj2->GetId()
         << " being merged to obj" << obj1->GetId()
         << " oml:" << arec->GetObjMergeLogprob()
         << " cdl:" << arec->GetClassDeltaLogprob()
         << " mp:" << arec->GetPriority() << endl;
  }
  // release obj2
  objects.erase(obj2->GetId());
  delete obj2;
  arec->SetObj2(NULL);
}


ObjectSegmenter::~ObjectSegmenter() {
  for (unordered_map<size_t, Object*>::iterator iter = objects.begin();
       iter != objects.end(); iter++) {
    delete iter->second;
  }
  for (vector<AdjacencyRecord*>::iterator iter = adjacency_space.begin();
       iter != adjacency_space.end(); iter++) {
    delete *iter;
  }
}


extern "C"
void c_run_segmentation(float* class_pred, int class_dim,
                        float* adj_pred, int offset_dim,
                        int img_width, int img_height,
                        int num_classes,
                        int* offset_list,
                        int* output,
                        int* object_class,
                        float same_different_bias,
                        float object_merge_factor,
                        float merge_logprob_bias) {
  ObjectSegmenterOption opt(same_different_bias,
                            object_merge_factor,
                            merge_logprob_bias);
  ObjectSegmenter segmenter(class_pred, class_dim,
                  adj_pred, offset_dim,
                  img_width, img_height, num_classes,
                  offset_list,
                  output,
                  object_class,
                  opt,
                  0);
  segmenter.RunSegmentation();
}
