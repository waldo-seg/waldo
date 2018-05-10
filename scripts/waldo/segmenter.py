
# Copyright      2018  Hossein Hadian

# Apache 2.0

""" This module contains codes and algorithms for post-processing the output of the nnet to
    find the objects in the image.
"""

import os
import sys
from heapq import heappush, heappop
import numpy as np
import warnings
import resource
import scipy.misc

class ObjPair:
    """
    This class is used to make an unordered pair from 2 Objects.
    Used for search and comparison. """

    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        if obj1.id > obj2.id:  # swap them
            self.obj1, self.obj2 = self.obj2, self.obj1

    def __hash__(self):
        assert(self.obj1.id <= self.obj2.id)
        return hash((self.obj1.id, self.obj2.id))

    def __eq__(self, other):
        return (self.obj1.id, self.obj2.id) == (other.obj1.id, other.obj2.id)

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "<ObjPair({},{})>".format(self.obj1.id, self.obj2.id)


class Object:
    """
    This class represents an "object" in the output image.
    """

    def __init__(self, pixels, id, segmenter):
        self.pixels = pixels
        self.compute_class_logprobs(segmenter)
        self.object_class = np.argmax(self.class_logprobs)
        # it's actually a map (from obj pairs to adj record) for faster search and access
        self.adjacency_list = {}
        self.id = id

    def compute_class_logprobs(self, segmenter):
        self.class_logprobs = np.zeros(segmenter.num_classes)
        for c in range(len(self.class_logprobs)):
            for p in self.pixels:
                self.class_logprobs[c] += segmenter.get_class_logprob(p, c)

    def class_logprob(self):
        return self.class_logprobs[self.object_class]

    def print(self):
        print("Object {}. Adj list:".format(self.id))
        for obj_pair in self.adjacency_list:
            print("\t{}   -->   {}".format(obj_pair, self.adjacency_list[obj_pair]))
        print("")
        print("Pixel list: {}".format(self.pixels))


    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "<OBJ:{} class:{} npix:{}  nadj:{}>".format(self.id, self.object_class, len(self.pixels), len(self.adjacency_list))


class AdjacencyRecord:
    def __init__(self, obj1, obj2, segmenter):
        self.obj1 = obj1
        self.obj2 = obj2
        self.compute_obj_merge_logprob(segmenter)
        if self.obj_merge_logprob is None:
            raise Exception(
                "Bad adjacency record. The given objects are not adjacent.")
        self.class_delta_logprob = None
        self.merged_class = None
        self.merge_priority = None
        self.update_merge_priority()


    def compute_obj_merge_logprob(self, segmenter, verbose=0):
        logprob = 0
        self.sameness_logprob = 0
        for o, i in zip(segmenter.offsets, range(len(segmenter.offsets))):
            for p1 in self.obj1.pixels:
                p2 = (p1[0] + o[0], p1[1] + o[1])
                if p2 in self.obj2.pixels:
                    same_prob = segmenter.get_sameness_prob(p1, i)
                    if verbose == 1: print("{}  {}    --->    {:0.3f}".format(p1, p2, np.log(1.0 - same_prob)))
                    logprob += np.log(same_prob) - np.log(1.0 - same_prob)

            for p1 in self.obj2.pixels:
                p2 = (p1[0] + o[0], p1[1] + o[1])
                if p2 in self.obj1.pixels:
                    same_prob = segmenter.get_sameness_prob(p1, i)
                    if verbose == 1: print("{}  {}    --->    {:0.3f}".format(p1, p2, np.log(1.0 - same_prob)))
                    logprob += np.log(same_prob) - np.log(1.0 - same_prob)
        self.obj_merge_logprob = logprob


    def compute_class_delta_logprob(self):
        if self.obj1.object_class == self.obj2.object_class:
            self.class_delta_logprob, self.merged_class = 0.0, self.obj1.object_class
        else:
            joint_class_logprobs = self.obj1.class_logprobs + self.obj2.class_logprobs
            self.merged_class = np.argmax(joint_class_logprobs)
            merged_class_joint_logprob = joint_class_logprobs[self.merged_class]
            self.class_delta_logprob = merged_class_joint_logprob - \
                            self.obj1.class_logprob() - self.obj2.class_logprob()

    def update_merge_priority(self):
        self.compute_class_delta_logprob()
        den = (len(self.obj1.pixels)* len(self.obj2.pixels))
        self.merge_priority = (self.obj_merge_logprob/2.0 +
                               self.class_delta_logprob) / den

    def obj_pair(self):
        return ObjPair(self.obj1, self.obj2)


    def print(self):
        print("Printing arec {}:".format(self))
        self.obj1.print()
        self.obj2.print()

    def __eq__(self, other):
        return self.obj_pair() == other.obj_pair()

    def __lt__(self, other):
        return self.merge_priority < other.merge_priority

    def __str__(self):
        return "<AREC-{}:  [{}, {}]  oml:{:0.2f}  cdl:{:0.2f}  mp:{:0.2f}>".format(
            id(self), self.obj1, self.obj2, self.obj_merge_logprob, self.class_delta_logprob, self.merge_priority)


class ObjectSegmenter:
    def __init__(self, nnet_class_probs, nnet_sameness_probs, num_classes, offsets):
        self.class_probs = nnet_class_probs
        self.sameness_probs = nnet_sameness_probs
        self.num_classes = num_classes
        self.offsets = offsets  # should be a list of tuples
        # the pixels here are python tuples (x,y) not numpy arrays
        self.pixel2obj = {}
        class_dim, img_width, img_height = self.class_probs.shape
        offset_dim, img_width, img_height = self.sameness_probs.shape
        assert(class_dim == self.num_classes)
        assert(offset_dim == len(self.offsets))
        self.img_width = img_width
        self.img_height = img_height

        self.objects = {}
        self.adjacency_records = {}
        self.queue = []   # Python's heapq
        self.init_objects_and_adjacency_records()
        self.show_stats()

    def init_objects_and_adjacency_records(self):
        print("Initializing the segmenter...")
        print("Max mem: {} GB".format(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
        obj_id = 0
        for row in range(self.img_height):
            for col in range(self.img_width):
                pixels = [(row, col)]
                obj = Object(pixels, obj_id, self)
                self.objects[obj_id] = obj
                self.pixel2obj[(row, col)] = obj
                obj_id += 1

        for row in range(self.img_height):
            for col in range(self.img_width):
                obj1 = self.pixel2obj[(row, col)]
                for (i, j) in self.offsets:
                    if (0 <= row + i < self.img_width and
                            0 <= col + j < self.img_height):
                        obj2 = self.pixel2obj[(row + i, col + j)]
                        arec = AdjacencyRecord(obj1, obj2, self)
                        self.adjacency_records[arec.obj_pair()] = arec
                        obj1.adjacency_list[arec.obj_pair()] = arec
                        obj2.adjacency_list[arec.obj_pair()] = arec
                        if arec.merge_priority >= 0:
                            heappush(self.queue, (-arec.merge_priority, arec))

    def get_class_logprob(self, pixel, class_index):
        assert(class_index < self.num_classes)
        return np.log(self.class_probs[class_index, pixel[0], pixel[1]])

    def get_sameness_prob(self, pixel, offset_index):
        assert(offset_index < len(self.offsets))
        return self.sameness_probs[offset_index, pixel[0], pixel[1]]

    def show_stats(self):
        print("Total number of objects: {}".format(len(self.objects)))
        print("Total number of adjacency records: {}".format(
            len(self.adjacency_records)))
        print("Total number of records in the queue: {}".format(len(self.queue)))
        pixperobj = sorted([len(obj.pixels)
                            for obj in self.objects.values()], reverse=True)
        print("Top 10 biggest objs (#pixels): {}".format(pixperobj[:10]))
        adjlistsize = sorted([len(obj.adjacency_list)
                              for obj in self.objects.values()], reverse=True)
        print("Top 10 biggest objs (adj_list size): {}".format(
            adjlistsize[:10]))

    def compute_total_logprob(self):
        tot_class_logprob = 0
        for obj in self.objects.values():
            tot_class_logprob += obj.class_logprob()


    def visualize(self, iter):
        img = np.zeros((self.img_height, self.img_width))
        k = 1
        for obj in self.objects.values():
            for p in obj.pixels:
                img[p] = k
            center = tuple(np.array(obj.pixels).mean(axis=0))
            img[int(center[0]), int(center[1])] = 0.0
            k += 1
        scipy.misc.imsave('{}.png'.format(iter), img)

    def debug(self):
        """
        Do some sanity checks """

        ## check if the current set of objects excatly cover the whole image
        pix2count = np.zeros((self.img_height, self.img_width))
        for obj in self.objects.values():
            for p in obj.pixels:
                pix2count[p] += 1
        if not (pix2count == 1).all():
            print("Error: pixels are not all covered or they are double counted", file=sys.stderr)
            np.set_printoptions(threshold=20000)
            print(pix2count)
            sys.exit(1)

        ## check the adjacency lists of the objects
        tot_obj_adj_records = 0
        for obj in self.objects.values():
            tot_obj_adj_records += len(obj.adjacency_list)
            for arec in obj.adjacency_list.values():
                assert(arec.obj_pair() in self.adjacency_records)
                assert((arec.obj1 is obj) ^ (arec.obj2 is obj))
        assert(tot_obj_adj_records == 2 * len(self.adjacency_records))


    def run_segmentation(self):
        print("Starting segmentation...")
        n = 0
        N = 500000  # max iters -- for experimentation
        target_objs = 4  # for experimentation
        self.verbose = 0
        while self.queue:
            if len(self.objects) <= target_objs:
                print("Target objects reached: {}".format(target_objs))
                break
            if len(self.queue) < 100:  # in case we want to see a few last steps of the algorithm
                self.verbose = 1
            if n > 100:
                self.verbose = 0
            if n % 5000 == 0:
                print("At iteration {}:  max mem: {:0.2f} GB".format(
                    n, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
                self.show_stats()
                print("")
                #self.visualize(n)
            if n > N:
                print("Breaking after {} iters.".format(N))
                break
            n += 1
            merge_cost, arec = heappop(self.queue)
            if self.verbose >= 1:
                print("Iter: {} Popped: {:0.2f},{}".format(n, merge_cost, arec), file=sys.stderr)
            merge_priority = -merge_cost
            if merge_priority != arec.merge_priority:
                if self.verbose >= 1:
                    print("Not merging {:0.2f} != {:0.2f}\n".format(
                        merge_priority, arec.merge_priority), file=sys.stderr)
                continue
            arec.update_merge_priority()
            if arec.merge_priority >= merge_priority:
                if self.verbose >= 1:
                    print("Merging...{:0.2f} >= {:0.2f}\n".format(
                        arec.merge_priority, merge_priority), file=sys.stderr)
                self.merge(arec)
                if np.random.random() > 0.999: self.debug()
            elif arec.merge_priority >= 0.0:
                if self.verbose >= 1:
                    print("Pushing with new mp: {:0.2f}\n".format(
                        arec.merge_priority), file=sys.stderr)
                heappush(self.queue, (-arec.merge_priority, arec))
            else:
                if self.verbose >= 1:
                    print("Not merging <0   {}\n".format(arec), file=sys.stderr)

        if len(self.queue) == 0:
            print("Finished. Queue is empty.")

        self.show_stats()
        self.visualize('final')

    def merge(self, arec):
        obj1, obj2 = arec.obj1, arec.obj2
        if obj1.id not in self.objects or obj2.id not in self.objects:
            return
        if obj1 is obj2:
            return
        if len(obj2.pixels) > len(obj1.pixels):  # swap them
            obj1, obj2 = obj2, obj1

#        vvv = False
#        if arec.obj1.id == 3498 or arec.obj1.id == 3499 or arec.obj2.id == 3499 or arec.obj2.id == 3498:
#            print("98-99-related: merging {} ::".format(arec))
#            arec.print()
#            vvv = True

#        if arec.obj1.id == 3497 and arec.obj2.id == 3498:
#            print("Merging the doomed {}:".format(arec))
#            arec.print()
        # now we are sure that obj1 has equal/more pixels
        obj1.object_class = arec.merged_class
        obj1.pixels += obj2.pixels
        obj1.class_logprobs += obj2.class_logprobs
        del self.adjacency_records[arec.obj_pair()]
        del obj1.adjacency_list[arec.obj_pair()]
        del obj2.adjacency_list[arec.obj_pair()]
        for this_arec in obj2.adjacency_list.values():
#            if vvv: print("this_arec: {}".format(this_arec))
            obj3 = this_arec.obj2 if this_arec.obj1 is obj2 else this_arec.obj1  # obj3 is any object adjacent to obj2 (never is obj1)
            assert(obj3 is not obj1)
#            if vvv: print("obj3: {}".format(obj3))
            del obj3.adjacency_list[this_arec.obj_pair()]
            del self.adjacency_records[this_arec.obj_pair()]

            if this_arec.obj1 is obj2:
                this_arec.obj1 = obj1
            if this_arec.obj2 is obj2:
                this_arec.obj2 = obj1
            obj3.adjacency_list[this_arec.obj_pair()] = this_arec
            self.adjacency_records[this_arec.obj_pair()] = this_arec
#            if vvv: print("this_arec NOWWWW: {}".format(this_arec))

            assert(this_arec.obj1 is not this_arec.obj2)

            if this_arec.obj_pair() in obj1.adjacency_list:
                that_arec = obj1.adjacency_list[this_arec.obj_pair()]
#                if vvv: print("WAAAAAAS in obj1.adjacency_list: {}".format(that_arec))
                that_arec.obj_merge_logprob += this_arec.obj_merge_logprob
#                if vvv: print("now that_arec: {}".format(that_arec))
                obj3.adjacency_list[that_arec.obj_pair()] = that_arec
                that_arec.update_merge_priority()
                if that_arec.merge_priority >= 0:
                    heappush(self.queue, (-that_arec.merge_priority, that_arec))
            else:
#                if vvv: print("was NOOOOOOOOOOOT: {}".format(this_arec))
                obj1.adjacency_list[this_arec.obj_pair()] = this_arec
#                obj3.adjacency_list[this_arec.obj_pair()] = this_arec
                old_obj_merge_logprob = this_arec.obj_merge_logprob
#                this_arec.compute_obj_merge_logprob(self)
#                if np.abs(this_arec.obj_merge_logprob - old_obj_merge_logprob) > 0.001:
#                    print("Error while merging {}".format(arec))
#                    print("old obj logprob: {}   new: {}".format(old_obj_merge_logprob, this_arec.obj_merge_logprob))
#                    this_arec.print()
#                    this_arec.compute_obj_merge_logprob(self, verbose=1)
#                    sys.exit(1)
                this_arec.update_merge_priority()
                if this_arec.merge_priority >= 0:
                    heappush(self.queue, (-this_arec.merge_priority, this_arec))
        if self.verbose >= 1:
            print("Deleting {} being merged to {} according to {}".format(obj2, obj1, arec), file=sys.stderr)
        if self.verbose >= 1:
            print("Adj list of obj {}:".format(obj1.id), file=sys.stderr)
            for obj_pair in obj1.adjacency_list:
                print("\t{}   -->   {}".format(obj_pair, obj1.adjacency_list[obj_pair]), file=sys.stderr)
            print("", file=sys.stderr)
        del self.objects[obj2.id]
