# Copyright      2018  Johns Hopkins University (author: Yiwen Shao)

# Apache 2.0


class UnetConfig:
    """
    A class to store certain configuration information that is needed
    to initialize a Unet. Making these network specific configurations
    store on disk can help us keep the network compatible in testing.
    """

    def __init__(self):
        """
        Initialize default network config values.
        """

        # number of subsampling/upsampling blocks in Unet
        self.depth = 5

        # number of output channels of the first cnn layer
        self.start_filters = 64

        # the way how upsampling are done in Unet.
        self.up_mode = 'transpose'

        # the way how feature maps with same size in downsampling and upsampling
        # are connected.
        self.merge_mode = 'concat'

    def validate(self, train_image_size):
        """
        Validate that configuration values are sensible.  Dies on error.
        """
        # the network can't downsample the input image to a size samller than 1*1
        assert (train_image_size >= 2 ** self.depth and
                train_image_size % (2 ** self.depth) == 0)
        assert self.up_mode in ['transpose', 'upsample']
        assert self.merge_mode in ['concat', 'add']

    # write the configuration file to 'filename'
    def write(self, filename):
        try:
            f = open(filename, 'w')
        except:
            raise Exception(
                "Failed to open file {0} for writing configuration".format(filename))

        for s in ['depth', 'start_filters', 'up_mode', 'merge_mode']:
            print("{0} {1}".format(s, self.__dict__[s]), file=f)
        f.close()

    def read(self, filename, train_image_size):
        try:
            f = open(filename, 'r')
        except:
            raise Exception(
                "Failed to open file {0} for reading configuration".format(filename))

        for line in f:
            a = line.split()
            if len(a) == 0 or a[0][0] == '#':
                continue
            if a[0] in ['depth', 'start_filters']:
                # parsing line like: 'num_classes 10'
                try:
                    self.__dict__[a[0]] = int(a[1])
                except:
                    raise Exception("Parsing config line in {0}: bad line {1}".format(
                        filename, line))
            elif a[0] == 'up_mode' or a[0] == 'merge_mode':
                try:
                    self.__dict__[a[0]] = a[1]
                except:
                    raise Exception("Parsing config line in {0}: bad line: {1}".format(
                        filename, line))
        self.validate(train_image_size)


def test():
    # very non-thorough test.
    n_c = UnetConfig()

    n_c.write('foo')
    n_c.read('foo', 128)
    n_c.write('foo')


# run this as: python3 unet_config.py to run the test code.
if __name__ == "__main__":
    test()

