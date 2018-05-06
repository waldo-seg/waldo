##
import os
import sys
from heapq import heappush, heappop
import numpy as np
import warnings
import resource
import scipy.misc

# offset_list = [(1, 1), (0, -2)]

def compute_class_logprobs(pixels, segmenter):
    logprobs = np.zeros(segmenter.num_classes)
    for c in range(len(logprobs)):
        for p in pixels:
            logprobs[c] += segmenter.get_class_logprob(p, c)
    return logprobs


def compute_obj_merge_logprob(obj1, obj2, segmenter):
    logprob = 0
    for o, i in zip(segmenter.offsets, range(len(segmenter.offsets))):
        for p1 in obj1.pixels:
            p2 = (p1[0]+o[0], p1[1]+o[1])
            if p2 in obj2.pixels:
                same_prob = segmenter.get_sameness_prob(p1, i)
                logprob += np.log(same_prob) - np.log(1.0 - same_prob)

        for p1 in obj2.pixels:
            p2 = (p1[0]+o[0], p1[1]+o[1])
            if p2 in obj1.pixels:
                same_prob = segmenter.get_sameness_prob(p1, i)
                logprob += np.log(same_prob) - np.log(1.0 - same_prob)
    return logprob

def compute_class_delta_logprob(adj_rec):
    if adj_rec.obj1.object_class == adj_rec.obj2.object_class:
        return 0.0, adj_rec.obj1.object_class
    else:
        joint_class_logprobs = adj_rec.obj1.class_logprobs + adj_rec.obj2.class_logprobs
        merged_class = np.argmax(joint_class_logprobs)
        merged_class_joint_logprob = joint_class_logprobs[merged_class]
        delta_logprob = merged_class_joint_logprob - adj_rec.obj1.class_logprob() - adj_rec.obj2.class_logprob()
    return delta_logprob, merged_class


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


class Object:
    """
    This class represents an "object" in the output image.
    """

    def __init__(self, pixels, id, segmenter):
        self.pixels = pixels
        self.class_logprobs = compute_class_logprobs(pixels, segmenter)
        self.object_class = np.argmax(self.class_logprobs)
        self.adjacency_list = {}   # it's actually a map (from obj pairs to adj record) for faster search and access
        self.id = id

    def class_logprob(self):
        return self.class_logprobs[self.object_class]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "<OBJ: {} class:{} npix: {}  nadj: {}>".format(self.id, self.object_class, len(self.pixels), len(self.adjacency_list))


class AdjacencyRecord:
    def __init__(self, obj1, obj2, segmenter):
        self.obj1 = obj1
        self.obj2 = obj2
        self.obj_merge_logprob = compute_obj_merge_logprob(obj1, obj2, segmenter)
        if self.obj_merge_logprob is None:
            raise Exception("Bad adjacency record. The given objects are not adjacent.")
        self.class_delta_logprob = None
        self.merged_class = None
        self.merge_priority = None
        self.update_merge_priority()

    def update_merge_priority(self):
        self.class_delta_logprob, self.merged_class = compute_class_delta_logprob(self)
        den = min(len(self.obj1.pixels), len(self.obj2.pixels))
        self.merge_priority = (self.obj_merge_logprob + self.class_delta_logprob) / den

    def __eq__(self, other):
        return ObjPair(self.obj1, self.obj2) == ObjPair(other.obj1, other.obj2)

    def __lt__(self, other):
        return self.merge_priority < other.merge_priority

    def __str__(self):
        return "<AREC-{}:  [{}, {}]  oml:{}  cdl:{}  mp:{}>".format(
            id(self), self.obj1, self.obj2, self.obj_merge_logprob, self.class_delta_logprob, self.merge_priority)

class ObjectSegmenter:
    def __init__(self, nnet_class_probs, nnet_sameness_probs, num_classes, offsets):
        self.class_probs = nnet_class_probs
        self.sameness_probs = nnet_sameness_probs
        self.num_classes = num_classes
        self.offsets = offsets  # should be a list of tuples
        self.pixel2obj = {}     # the pixels here are python tuples (x,y) not numpy arrays
        class_dim, img_width, img_height = self.class_probs.shape
        offset_dim, img_width, img_height = self.sameness_probs.shape
        assert(class_dim == self.num_classes)
        assert(offset_dim == len(self.offsets))
        self.img_width = img_width
        self.img_height = img_height

        self.objects = {}
        self.adjacency_records = []
        self.queue = []   # Python's heapq
        self.init_objects_and_adjacency_records()
        self.show_stats()

    def init_objects_and_adjacency_records(self):
        print("Initializing the segmenter...")
        print("Mem: {} GB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024))
        obj_id = 0
        for row in range(self.img_height):
            for col in range(self.img_width):
                # pixels = np.array([(x, y)])
                # np.append(p, [[5,6]], axis=0)
                pixels = [(row, col)]
                obj = Object(pixels, obj_id, self)
                obj_id += 1
                self.objects[obj_id] = obj
                self.pixel2obj[(row, col)] = obj

        for row in range(self.img_height):
            for col in range(self.img_width):
                this_pixel = np.array([row, col])
                obj1 = self.pixel2obj[tuple(this_pixel)]
                assert(obj1 is not None)
                for o in self.offsets:
                    other_pixel = this_pixel + np.array(o)
                    if (other_pixel < [self.img_width, self.img_height]).all() and (other_pixel >= [0, 0]).all():
                        obj2 = self.pixel2obj[tuple(other_pixel)]
                        assert(obj2 is not None)
                        arec = AdjacencyRecord(obj1, obj2, self)
                        self.adjacency_records.append(arec)
                        obj1.adjacency_list[ObjPair(obj1, obj2)] = arec
                        obj2.adjacency_list[ObjPair(obj1, obj2)] = arec
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
        print("Total number of adjacency records: {}".format(len(self.adjacency_records)))
        print("Total number of records in the queue: {}".format(len(self.queue)))
        pixperobj = sorted([len(obj.pixels) for obj in self.objects.values()], reverse=True)
        print("top 10 biggest objs: {}".format(pixperobj[:10]))
        adjlistsize = sorted([len(obj.adjacency_list) for obj in self.objects.values()], reverse=True)
        print("top 10 biggest objs ito adjlistsize: {}".format(adjlistsize[:10]))

    def visualize(self):
        pix2class = np.zeros((self.img_width, self.img_height))
        for obj in self.objects.values():
            for p in obj.pixels:
                pix2class[p] = obj.object_class
        scipy.misc.imsave('final.png', pix2class)

    def run_segmentation(self):
        print("Starting segmentation...")
        n = 0
        N = 200000  # max iters -- for experimentation
        target_objs = 10  # for experimentation
        verbose = 0
        while self.queue:
            if len(self.objects) <= target_objs:
                print("Target objects reached: {}".format(target_objs))
                break
            if len(self.queue) < 500:
                verbose = 1
            n += 1
            if n % 1000 == 0:
                print("At iteration {}:   mem: {} GB".format(n, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024))
                self.show_stats()
                print("")
            if n > N:
                print("Breaking after {} iters.".format(N))
                break;
            merge_cost, arec = heappop(self.queue)
            if verbose>=1: print("Popped: {},{}".format(merge_cost, arec), file=sys.stderr)
            merge_priority = -merge_cost
            if merge_priority != arec.merge_priority:
                if verbose>=1: print("Not merging {} != {}".format(merge_priority, arec.merge_priority), file=sys.stderr)
                continue
            arec.update_merge_priority()
            if arec.merge_priority >= merge_priority:
                if verbose>=1: print("Merging...{} >= {}".format(arec.merge_priority, merge_priority), file=sys.stderr)
                self.merge(arec)
            elif arec.merge_priority >= 0.0:
                if verbose>=1: print("Pushing with new mp: {}".format(arec.merge_priority), file=sys.stderr)
                heappush(self.queue, (-arec.merge_priority, arec))
            else:
                if verbose>=1: print("Not merging >=0   {}".format(arec), file=sys.stderr)
            if verbose>=1: print("", file=sys.stderr)

        print("Finished. queue is empty.")
        self.visualize()

    def merge(self, arec):
        obj1, obj2 = arec.obj1, arec.obj2
        if obj1.id not in self.objects or obj2.id not in self.objects:
            return
        if obj1 is obj2:
            return
        if len(obj2.pixels) > len(obj1.pixels):  # swap them
            obj1, obj2 = obj2, obj1
        # now we are sure that obj1 has equal/more pixels
        obj1.object_class = arec.merged_class
        obj1.pixels += obj2.pixels
        obj1.class_logprobs += obj2.class_logprobs
        for this_arec in obj2.adjacency_list.values():
            needs_update = False
            #if this_arec == arec:
            #    continue
            if this_arec.obj1 is obj2:
                this_arec.obj1 = obj1
                needs_update = True
            if this_arec.obj2 is obj2:
                this_arec.obj2 = obj1
                needs_update = True
            if  this_arec.obj1 is this_arec.obj2:
                continue
            if ObjPair(this_arec.obj1, this_arec.obj2) in obj1.adjacency_list:
                that_arec = obj1.adjacency_list[ObjPair(this_arec.obj1, this_arec.obj2)]
                if that_arec == this_arec:
                    continue
                that_arec.obj_merge_logprob += this_arec.obj_merge_logprob
                that_arec.update_merge_priority()
                if that_arec.merge_priority >= 0:
                    #print("ta-Pushing {}".format(that_arec), file=sys.stderr)
                    heappush(self.queue, (-that_arec.merge_priority, that_arec))
            else:
                obj1.adjacency_list[ObjPair(this_arec.obj1, this_arec.obj2)] = this_arec
            if needs_update:
                this_arec.update_merge_priority()
                if this_arec.merge_priority >= 0:
                    #print("nu-Pushing {}".format(this_arec), file=sys.stderr)
                    heappush(self.queue, (-this_arec.merge_priority, this_arec))
        #print("Deleting {} being merged to {} according to {}".format(obj2, obj1, arec), file=sys.stderr)
        del self.objects[obj2.id]
        del obj2
