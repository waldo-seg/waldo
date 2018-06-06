# Copyright      2018  Daniel Povey, Hossein Hadian

# Apache 2.0

""" This module contains codes and algorithms for post-processing the output of
    the nnet to find the objects in the image.
    Specifically it is a greedy algorithm in which the only operation is merging
    objects, starting from individual pixels, and the only choice is in which
    order to merge objects.  At all stages of optimization, objects will maintain
    their optimal class assignment.
"""

import os
import sys
from heapq import heappush, heappop
import numpy as np
import warnings
import resource
import scipy.misc
from collections import namedtuple

SegmenterOptions = namedtuple('SegmenterOptions',
                              ['same_different_bias',
                               'object_merge_factor',
                               'merge_logprob_bias'])


class Object:
    """
    This class represents an "object" in the output image.
    Attributes:
        object_class:    A record of the current assigned class (an integer)
        pixels:    A set of pixels (2-tuples) that are part of the object
        class_logprobs:    An array indexed by class, of the total (over all pixels
                in this object) of the log-prob of assigning this pixel to this class;
                the object_class corresponds to the index of the max element of this.
        adjacency_list:   A list of adjacency records, recording other objects to which
                this object is adjacent.  ('Adjacent' means "linked by an offset", not
                adjacency in the normal sense). It's actually a map (from obj
                pairs to adjacency record) for faster search and access.
    """

    def __init__(self, pixels, id, segmenter):
        self.pixels = pixels
        self.compute_class_logprobs(segmenter)
        self.object_class = np.argmax(self.class_logprobs)
        self.adjacency_list = {}
        self.id = id
        self.sameness_logprob = 0

    def compute_class_logprobs(self, segmenter):
        self.class_logprobs = np.zeros(segmenter.num_classes)
        for c in range(len(self.class_logprobs)):
            for p in self.pixels:
                self.class_logprobs[c] += segmenter.get_class_logprob(p, c)

    def class_logprob(self):
        return self.class_logprobs[self.object_class]

    def print(self):
        print("Object {}. Adj list:".format(self))
        for obj_pair in self.adjacency_list:
            print("\t{}   -->   {}".format(obj_pair,
                                           self.adjacency_list[obj_pair]))
        print("")
        print("Pixel list: {}".format(self.pixels))

    def compute_sameness_logprob(self, segmenter):
        """ This is only used for debugging purposes. """
        self.sameness_logprob = 0
        for i, o in enumerate(segmenter.offsets):
            for p1 in self.pixels:
                p2 = (p1[0] + o[0], p1[1] + o[1])
                if p2 in self.pixels:
                    same_prob = segmenter.get_sameness_prob(p1, i)
                    self.sameness_logprob += np.log(same_prob)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "<OBJ:{} class:{} npix:{}  nadj:{}>".format(self.id,
                                                           self.object_class,
                                                           len(self.pixels),
                                                           len(self.adjacency_list))


class AdjacencyRecord:
    """
    This class implements an adjacency record with functions for computing object merge
    log-probs and class merge log-probs.
    Attributes:
        obj1, obj2:    The two objects to which it refers
        object_merge_logprob:    This is the change in log-probability from merging these two objects,
              without considering the effect arising from changes of class assignments.
              This is the sum of the following:
              For each p,o such that o is in "offsets", p is in one of the two objects
              and p+o is in the other, the value log(p(same) / p(different)), i.e.g
              log(b_{p,o} / (1-b{p,o})).
              Note: if the sum above had no terms in it, this
              adjacency record should not exist because the objects would not be
              "adjacent" in the sense which we refer to.
        merge_priority:    This merge priority is a heuristic which will determine what kinds of
              objects will get merged first, and is a key choice that we'll have to
              experiment with.  (Note: you can change the sign if it turns out to be
              easier for python heap reasons).  The general idea will be:
              merge_priority = merge_log_prob / den
              where merge_log_prob is the log-prob change from doing this merge, and
              for example, "den" might be the maximum of the num-pixels in
              object1 and object2.  We can experiment with different heuristics for
              "den" though.
        class_delta_logprob:   It is a term representing a change in the total
              log-prob that we'll get from merging the two objects, that arises from
              forcing the class assignments to be the same.  If the two objects already
              have the same assigned class, this will be zero.  If different, then this
              is a value <= 0 which can be obtained by summing the objects'
              'class_logprobs' arrays, finding the largest log-prob in the total, and
              subtracting the total from the current class-assignments of the two
              objects.
        merged_class:    The class that the merged object would have, obtained when figuring
              out class_delta_log_prob.
    """

    def __init__(self, obj1, obj2, segmenter, pixel=None, offset=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.sort_and_update_hash()
        if pixel is not None and offset is not None:
           same_prob = segmenter.get_sameness_prob(pixel, offset)
           log_same_prob = np.log(same_prob)
           log_different_prob = np.log(1.0 - same_prob)
           self.differentness_logprob = log_different_prob
           self.sameness_logprob = log_same_prob
           self.obj_merge_logprob = log_same_prob - log_different_prob
        else:
            self.compute_obj_merge_logprob(segmenter)
            if self.obj_merge_logprob is None:
               raise Exception(
                     "Bad adjacency record. The given objects are not adjacent.")
        self.class_delta_logprob = None
        self.merged_class = None
        self.merge_priority = None
        self.update_merge_priority(segmenter)

    def compute_obj_merge_logprob(self, segmenter):
        logprob = 0
        adjacent = False
        self.differentness_logprob = 0
        self.sameness_logprob = 0
        for o, i in zip(segmenter.offsets, range(len(segmenter.offsets))):
            for p1 in self.obj1.pixels:
                p2 = (p1[0] + o[0], p1[1] + o[1])
                if p2 in self.obj2.pixels:
                    adjacent = True
                    same_prob = segmenter.get_sameness_prob(p1, i)
                    log_same_prob = np.log(same_prob)
                    log_different_prob = np.log(1.0 - same_prob)
                    self.differentness_logprob += log_different_prob
                    self.sameness_logprob += log_same_prob
                    logprob += log_same_prob - log_different_prob

            for p1 in self.obj2.pixels:
                p2 = (p1[0] + o[0], p1[1] + o[1])
                if p2 in self.obj1.pixels:
                    adjacent = True
                    same_prob = segmenter.get_sameness_prob(p1, i)
                    log_same_prob = np.log(same_prob)
                    log_different_prob = np.log(1.0 - same_prob)
                    self.differentness_logprob += log_different_prob
                    self.sameness_logprob += log_same_prob
                    logprob += log_same_prob - log_different_prob
        self.obj_merge_logprob = logprob if adjacent else None

    def compute_class_delta_logprob(self):
        if self.obj1.object_class == self.obj2.object_class:
            self.class_delta_logprob, self.merged_class = 0.0, self.obj1.object_class
        else:
            joint_class_logprobs = self.obj1.class_logprobs + self.obj2.class_logprobs
            self.merged_class = np.argmax(joint_class_logprobs)
            merged_class_joint_logprob = joint_class_logprobs[self.merged_class]
            self.class_delta_logprob = merged_class_joint_logprob - \
                self.obj1.class_logprob() - self.obj2.class_logprob()

    def update_merge_priority(self, segmenter):
        self.compute_class_delta_logprob()
        den = len(self.obj1.pixels) * len(self.obj2.pixels)
        self.merge_priority = (self.obj_merge_logprob * segmenter.opts.object_merge_factor +
                               self.class_delta_logprob + segmenter.opts.merge_logprob_bias) / den

    def obj_pair(self):
        return ObjPair(self.obj1, self.obj2)

    def sort_and_update_hash(self):
        if self.obj1.id > self.obj2.id:  # swap them
            self.obj1, self.obj2 = self.obj2, self.obj1
        self.cached_hash = hash((self.obj1.id, self.obj2.id))

    def __hash__(self):
        return self.cached_hash

    def __eq__(self, other):
        return (self.obj1.id, self.obj2.id) == (other.obj1.id, other.obj2.id)

    def __ne__(self, other):
        return not(self == other)

    def print(self):
        print("Objects in arec {}:".format(self))
        self.obj1.print()
        self.obj2.print()

    def __lt__(self, other):
        return self.merge_priority < other.merge_priority

    def __str__(self):
        return "<AREC-{}:  [{}, {}]  oml:{:0.2f}  cdl:{:0.2f}  mp:{:0.2f}>".format(
            id(self), self.obj1, self.obj2, self.obj_merge_logprob, self.class_delta_logprob, self.merge_priority)

class ObjectSegmenter:
    def __init__(self, nnet_class_probs, nnet_sameness_probs, num_classes,
                 offsets, opts=None):
        self.opts = opts
        if self.opts is None:
            self.opts = self.default_options()
        print(self.opts)
        epsilon = np.finfo(np.float32).eps
        self.class_probs = nnet_class_probs.clip(epsilon, 1.0 - epsilon)
        self.sameness_probs = nnet_sameness_probs.clip(epsilon, 1.0 - epsilon)
        if self.opts.same_different_bias != 0.0:
            sameness_probs_biased_logit = (np.log(self.sameness_probs) -
                                           np.log(1.0 - self.sameness_probs) +
                                           self.opts.same_different_bias)
            self.sameness_probs = 1.0 / \
                (1.0 + np.exp(-sameness_probs_biased_logit))
        self.num_classes = num_classes
        self.offsets = offsets  # should be a list of tuples
        # the pixels here are python tuples (x,y) not numpy arrays
        self.pixel2obj = {}
        class_dim, self.img_height, self.img_width = self.class_probs.shape
        offset_dim, img_height, img_width = self.sameness_probs.shape
        assert class_dim == self.num_classes
        assert offset_dim == len(self.offsets)
        assert self.img_height == img_height
        assert self.img_width == img_width

        self.objects = {}
        self.adjacency_records = {}
        self.queue = []   # Python's heapq
        self.init_objects_and_adjacency_records()


    def default_options(self):
        return SegmenterOptions(same_different_bias=0.0,
                                object_merge_factor=1.0,
                                merge_logprob_bias=0.0)

    def init_objects_and_adjacency_records(self):
        print("Initializing the segmenter...")
        print("Max mem: {} GB".format(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
        obj_id = 0
        for row in range(self.img_height):
            for col in range(self.img_width):
                pixels = set([(row, col)])
                obj = Object(pixels, obj_id, self)
                self.objects[obj_id] = obj
                self.pixel2obj[(row, col)] = obj
                obj_id += 1

        for row in range(self.img_height):
            for col in range(self.img_width):
                obj1 = self.pixel2obj[(row, col)]
                for o, idx in zip(self.offsets, range(len(self.offsets))):
                    (i, j) = o
                    if (0 <= row + i < self.img_height and
                            0 <= col + j < self.img_width):
                        obj2 = self.pixel2obj[(row + i, col + j)]
                        arec = AdjacencyRecord(obj1, obj2, self, (row, col), idx)
                        self.adjacency_records[arec] = arec
                        obj1.adjacency_list[arec] = arec
                        obj2.adjacency_list[arec] = arec
                        if arec.merge_priority >= 0:
                            heappush(self.queue, (-arec.merge_priority, arec))

    def get_class_logprob(self, pixel, class_index):
        return np.log(self.class_probs[class_index, pixel[0], pixel[1]])

    def get_sameness_prob(self, pixel, offset_index):
        return self.sameness_probs[offset_index, pixel[0], pixel[1]]

    def show_stats(self):
        print("Total logprob: "
              "{:.3f}".format(self.compute_total_logprob()))
        print("Total number of objects: {}".format(len(self.objects)))
        print("Total number of adjacency records: "
              "{}".format(len(self.adjacency_records)))
        print("Total number of records in the queue: {}".format(len(self.queue)))
        pixperobj = sorted([len(obj.pixels)
                            for obj in self.objects.values()], reverse=True)
        print("Top 10 biggest objs (#pixels): {}".format(pixperobj[:10]))
        adjlistsize = sorted([len(obj.adjacency_list)
                              for obj in self.objects.values()], reverse=True)
        print("Top 10 biggest objs (adj_list size): {}".format(
            adjlistsize[:10]))

    def compute_total_logprob_from_scratch(self):
        """ This is for debugging only. """
        tot_class_logprob = 0
        tot_differentness_logprob = 0
        tot_sameness_logprob = 0
        for obj in self.objects.values():
            for p in obj.pixels:
                self.pixel2obj[p] = obj
                tot_class_logprob += self.get_class_logprob(p, obj.object_class)
        for row in range(self.img_height):
            for col in range(self.img_width):
                p1 = (row, col)
                obj1 = self.pixel2obj[p1]
                for i, o in enumerate(self.offsets):
                    if (0 <= row + o[0] < self.img_height and
                            0 <= col + o[1] < self.img_width):
                        obj2 = self.pixel2obj[(row + o[0], col + o[1])]
                        if obj1 is obj2 or obj1 == obj2:
                            tot_sameness_logprob += np.log(
                                self.get_sameness_prob(p1, i))
                        else:
                            tot_differentness_logprob += np.log(
                                1.0 - self.get_sameness_prob(p1, i))
        return tot_class_logprob + (tot_differentness_logprob +
                                    tot_sameness_logprob) * self.opts.object_merge_factor


    def compute_total_logprob(self):
        tot_class_logprob = 0
        tot_differentness_logprob = 0
        tot_sameness_logprob = 0
        for obj in self.objects.values():
            tot_class_logprob += obj.class_logprob()
            tot_sameness_logprob += obj.sameness_logprob
        for arec in self.adjacency_records.values():
            tot_differentness_logprob += arec.differentness_logprob
        return tot_class_logprob + (tot_differentness_logprob +
                                    tot_sameness_logprob) * self.opts.object_merge_factor

    def visualize(self, iter):
        img = np.zeros((self.img_height, self.img_width))
        k = 1
        for obj in self.objects.values():
            for p in obj.pixels:
                img[p] = k
            center = tuple(np.array(list(obj.pixels)).mean(axis=0))
            img[int(center[0]), int(center[1])] = 0.0
            k += 1
        scipy.misc.imsave('{}.png'.format(iter), img)

    def prune(self, threshold=200.0):
        # Find the biggest background object:
        num_pixels = 0
        for obj in self.objects.values():
            if obj.object_class == 0 and len(obj.pixels) > num_pixels:
                background_obj = obj
                num_pixels = len(obj.pixels)

        objects_to_be_merged = []
        for obj in self.objects.values():
            nonbackground_score = obj.class_logprob() - obj.class_logprobs[0]
            if self.verbose >= 2:
                print("obj: {}   -->   {:0.2f}".format(len(obj.pixels), nonbackground_score))
            if nonbackground_score < threshold and obj is not background_obj:
                objects_to_be_merged.append(obj)

        for obj in objects_to_be_merged:
            if self.verbose >= 1:
                print("Merging obj with {} pixels to "
                      "background...".format(len(obj.pixels)))
            background_obj.pixels = background_obj.pixels.union(obj.pixels)
            del self.objects[obj.id]
        print("Pruned {} objects (merged into background). Final objects:"
              " {}".format(len(objects_to_be_merged), len(self.objects)))

    def output_mask(self):
        mask = np.zeros((self.img_height, self.img_width), dtype=int)
        k = 1
        object_class = []
        for obj in self.objects.values():
            # skip background object
            if obj.object_class == 0:
                continue
            object_class.append(obj.object_class)
            for p in obj.pixels:
                mask[p] = k
            k += 1
        return mask, object_class

    def debug(self):
        """
        Do some sanity checks and make sure certain quantities have values that
        they should have.
        This function is quite time-consuming and should not be called too
        many times."""

        # check if the current set of objects excatly cover the whole image
        pix2count = np.zeros((self.img_height, self.img_width))
        for obj in self.objects.values():
            for p in obj.pixels:
                pix2count[p] += 1
        if not (pix2count == 1).all():
            print("Error: pixels are not all covered or they are double counted")
            np.set_printoptions(threshold=20000)
            print(pix2count)
            sys.exit(1)

        # check the adjacency lists of the objects
        tot_obj_adj_records = 0
        for obj in self.objects.values():
            tot_obj_adj_records += len(obj.adjacency_list)
            for arec in obj.adjacency_list.values():
                assert arec in self.adjacency_records
                assert (arec.obj1 is obj) ^ (arec.obj2 is obj)
                # make sure that re-computing obj-mere-logprob does not change it
                # this is too costly to run, so only do it randomly with a small chance
                if np.random.random() > 0.95:
                    obj_merge_logprob = arec.obj_merge_logprob
                    arec.compute_obj_merge_logprob(self)
                    if np.abs(arec.obj_merge_logprob - obj_merge_logprob) > 0.001:
                        print("Error re-computing obj-merge logprob changed it for "
                              "arec {}".format(arec))
                        print("Old logprob: {}   "
                              "new logprob: {}".format(obj_merge_logprob,
                                                       arec.obj_merge_logprob))
                        arec.print()
                        sys.exit(1)

        assert tot_obj_adj_records == 2 * len(self.adjacency_records)

    def run_segmentation(self):
        """ This is the top-level function that performs the optimization.
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
        """
        print("Starting segmentation...")
        n = 0
        self.verbose = 0
        self.do_debugging = False
        while self.queue:
            if n % 500000 == 0:
                print("At iteration {}:  max mem: {:0.2f} GB".format(
                    n, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
                self.show_stats()
                if self.do_debugging:
                    print("Logprob from scratch: {}".format(
                        self.compute_total_logprob_from_scratch()))
                print("")
            n += 1
            merge_cost, arec = heappop(self.queue)
            merge_priority = -merge_cost
            if merge_priority != arec.merge_priority:
                continue
            arec.update_merge_priority(self)
            if arec.merge_priority >= merge_priority:
                self.merge(arec)
            elif arec.merge_priority >= 0:
                heappush(self.queue, (-arec.merge_priority, arec))

        if len(self.queue) == 0:
            print("Finished. Queue is empty.")

        self.show_stats()
        self.visualize('final')
        if self.verbose >= 1:
            print("Final logprob from scratch: {}".format(
                self.compute_total_logprob_from_scratch()))
        return self.output_mask()

    def merge(self, arec):
        """ This is the most nontrivial aspect of the algorithm: how to merge objects.
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
        """

        obj1, obj2 = arec.obj1, arec.obj2
        if obj1.id not in self.objects or obj2.id not in self.objects:
            return
        if obj1 is obj2:
            return
        if len(obj2.pixels) > len(obj1.pixels):
            obj1, obj2 = obj2, obj1

        assert np.abs(arec.obj_merge_logprob -
                      (arec.sameness_logprob - arec.differentness_logprob)) < 0.001

        if self.do_debugging:
            old_logprob = arec.obj_merge_logprob
            arec.compute_obj_merge_logprob(self)
            if np.abs(arec.obj_merge_logprob - old_logprob) > 0.001:
                print("Error: object merge logprob changed unexpectedly. "
                      "{}  !=  {}".format(arec.obj_merge_logprob, old_logprob))
                arec.print()
                sys.exit(1)

        # now we are sure that obj1 has equal/more pixels
        obj1.object_class = arec.merged_class
        obj1.pixels = obj1.pixels.union(obj2.pixels)
        obj1.class_logprobs += obj2.class_logprobs
        obj1.sameness_logprob += arec.sameness_logprob + obj2.sameness_logprob

        del self.adjacency_records[arec]
        del obj1.adjacency_list[arec]
        del obj2.adjacency_list[arec]
        for this_arec in obj2.adjacency_list.values():
            # obj3 is any object adjacent to obj2 (never is obj1):
            obj3 = this_arec.obj2 if this_arec.obj1 is obj2 else this_arec.obj1
            assert obj3 is not obj1

            del obj3.adjacency_list[this_arec]
            del self.adjacency_records[this_arec]
            if this_arec.obj1 is obj2:
                this_arec.obj1 = obj1
            if this_arec.obj2 is obj2:
                this_arec.obj2 = obj1
            this_arec.sort_and_update_hash()

            assert this_arec.obj1 is not this_arec.obj2

            if this_arec in obj1.adjacency_list:
                that_arec = obj1.adjacency_list[this_arec]
                that_arec.obj_merge_logprob += this_arec.obj_merge_logprob
                that_arec.differentness_logprob += this_arec.differentness_logprob
                that_arec.sameness_logprob += this_arec.sameness_logprob
                # make sure it is practically deleted from the queue
                this_arec.merge_priority = -100000.0
                self.adjacency_records[that_arec] = that_arec
                obj3.adjacency_list[that_arec] = that_arec
                that_arec.update_merge_priority(self)
                if that_arec.merge_priority >= 0:
                    heappush(self.queue, (-that_arec.merge_priority, that_arec))
            else:
                obj1.adjacency_list[this_arec] = this_arec
                obj3.adjacency_list[this_arec] = this_arec
                self.adjacency_records[this_arec] = this_arec
                this_arec.update_merge_priority(self)
                if this_arec.merge_priority >= 0:
                    heappush(self.queue, (-this_arec.merge_priority, this_arec))
        if self.verbose >= 2:
            print("Deleting {} being merged to {} according "
                  "to {}".format(obj2, obj1, arec))
        del self.objects[obj2.id]
