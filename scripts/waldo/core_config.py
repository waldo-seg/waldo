# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


class CoreConfig:
    """
    A class to store certain configuration information that is needed
    by core parts of Waldo, and read and write this information from
    a config file on disk.
    """

    def __init__(self):
        """
        Initialize default config values.  This is mainly just to
        illustrate reasonable defaults, as in practice we anticipate
        that we'll usually load this config from a file on disk
        after initializing.
        """

        # num_classes is the number of classes of object, e.g.  for text box
        # detection it might be 2 classes, with 0 == background, 1 == text-box.
        self.num_classes = 2

        # num_colors is the number of channels in the input images: e.g. 1 for
        # black and white, 3 for color.  We'll convert color images to rgb
        # representation (if num_colors == 3).
        self.num_colors = 1

        # 'offsets': a list of pairs.  For reference, this default offsets list
        # was produced by the following python code.
        ##!/usr/bin/env python3
        #import math
        #num_points = 10
        #size_ratio = 1.4
        #angle = math.pi * 5/9  # 100 degrees: just over 90 degrees.
        #offsets=[]
        #for n in range(num_points):
        #    x = round(math.cos(n * angle) * math.pow(size_ratio, n))
        #    y = round(math.sin(n * angle) * math.pow(size_ratio, n))
        #    offsets += [(x,y)]
        #print(offsets)
        self.offsets = [(1, 0), (0, 1), (-2, -1), (1, -2), (3, 2), (-4, 3), (-4, -7), (10, -4), (3, 15), (-21, 0)]

        # The amount of zero-padding we do around images prior to training and test.
        # (more zero padding than this will be done for images that are smaller than
        # self.train_image_size).
        self.padding = 10

        # The size of the parts of training images that we train on (in order to
        # form a fixed minibatch size).  These are derived from the input images
        # by padding and then random cropping.
        self.train_image_size = 256

    def validate(self):
        """
        Validate that configuration values are sensible.  Dies on error.
        """
        assert self.num_classes >= 2
        # We can change the assertion that num_colors <= 3 later on if we ever
        # need to operate on images with larger color spaces.
        assert self.num_colors >= 1 and self.num_colors <= 3

        assert len(self.offsets) > 0 and not (0,0) in self.offsets
        offsets_set = set(self.offsets)
        assert len(offsets_set) == len(self.offsets)
        for o in self.offsets:
            assert isinstance(o, tuple) and len(o) == 2
            assert 0 != (0,0)
            # check that the negative of an offset is not in the set-- that
            # would be redundant.
            assert not (-o[0],-o[1]) in offsets_set

        assert self.padding > 0
        assert self.train_image_size > 0 and self.train_image_size > 4 * self.padding


    # write the configuration file to 'filename'
    def write(self, filename):
        try:
            f = open(filename, 'w')
        except:
            raise Exception("Failed to open file {0} for writing configuration".format(filename))

        for s in [ 'num_classes', 'num_colors',
                   'padding', 'train_image_size' ]:
            print("{0} {1}".format(s, self.__dict__[s]), file=f)
        print("offsets {}".format('  '.join(['{0} {1}'.format(o[0],o[1]) for o in self.offsets])),
              file=f)
        f.close()

    def read(self, filename):
        try:
            f = open(filename, 'r')
        except:
            raise Exception("Failed to open file {0} for reading configuration".format(filename))

        for line in f:
            a = line.split()
            if len(a) == 0 or a[0][0] == '#':
                continue
            if len(a) == 2 and a[0] in [ 'num_classes', 'num_colors',
                                         'padding', 'train_image_size' ]:
                # parsing line like: 'num_classes 10'
                try:
                    self.__dict__[a[0]] = int(a[1])
                except:
                    raise Exception("Parsing config line in {0}: bad line {1}".format(
                        filename, line))
            elif a[0] == 'offsets':
                # parsing line like: 'offsets  1 0  0 -2   3 1'
                if len(a) < 5 or len(a) % 2 == 0:
                    raise Exception("Parsing offsets config line in {0}: bad num-fields: {1}".format(
                        filename, line))
                try:
                    num_offsets = (len(a) - 1) // 2
                    self.offsets = []
                    for i in range(num_offsets):
                        self.offsets.append((int(a[i*2 + 1]), int(a[i*2 + 2])))
                except:
                    raise Exception("Parsing offsets config line in {0}: bad offsets line: {1}".format(
                        filename, line))
        self.validate()


def test():
    # very non-thorough test.
    c = CoreConfig()
    c.write('foo')
    c.read('foo')
    c.write('foo')


# run this as: python3 core_config.py to run the test code.
if __name__ == "__main__":
    test()
