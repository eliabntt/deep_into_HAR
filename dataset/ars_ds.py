import numpy as np
import scipy.io

class ARSDataset():
    """Interface to the ARS dataset."""

    def __init__(self, path):
        self.path = path
        self._create_label_maps()

    def load_dataset(self, body_frame=True, flatten_labels=True, include_time=True):
        # skip processing if saved dataset is requested
        if 'npz' in self.path:
            return self._load_npz(self.path)

        # load dataset from matlab mat file
        self.dataset = scipy.io.loadmat(self.path)
        keys = [ i for i in sorted(self.dataset) if '__' not in i ]

        # collect data from all tests
        imu = np.empty((0,10 if include_time else 9))
        labels = np.empty((0,10 if include_time else 9))
        for k in keys:
            print('Loading {}:'.format(k).ljust(52,' '), end='')
            i,l = self._get_single_test_data(k, body_frame, flatten_labels, include_time)
            imu = np.append(imu, i, axis=0)
            labels = np.append(labels, l)
            print('{} elements'.format(i.shape[0]).rjust(15,' '))

        print('Data shape:  {}\nLabel shape: {}'.format(imu.shape, labels.shape))
        self.imu = imu
        self.labels = labels
        return imu, labels

    def _load_npz(self, path):
        d = np.load(path)
        self.imu = d['imu']
        self.labels = d['labels']
        return self.imu, self.labels

    def save_dataset(self, path):
        np.savez_compressed(path, {
            'imu': self.imu,
            'labels': self.labels
        })

    def _get_single_test_data(self, test_key, body_frame=True, flatten_labels=True, include_time=True):
        """
        Gets data of a single test from the dataset.

        test_key: the key representing the test to process
        body_frame: whether to convert coordinates to body frame, instead of using sensor frame
        flatten_labels: return labels as an array as long as returned sensor data, instead of an array of bounds
        include_time: whether to include or remove time column from sensor data
        """

        imu_data, cosine_matrices, activities, activities_bounds = self.dataset[test_key][0]

        # throwing away useless nested arrays
        activities = np.array([ act[0] for act in activities[0] ])
        activities_bounds = activities_bounds[0]

        # integrity checks on time and length
        assert([ imu_data[i][0] == cosine_matrices[i][0] for i in range(len(imu_data)) ])
        assert(len(activities_bounds) == 2*len(activities))

        # change labels to int numbers
        activities = np.array([ self.map_encode[a] for a in activities ])

        # get a single array of labels instead of labels + bounds
        if flatten_labels:
            activities_flat = self._reply_labels(activities, activities_bounds)
            assert(len(imu_data) == len(activities_flat))

        # convert from sensor frame to body frame
        if body_frame:
            for i, imu in enumerate(imu_data):
                imu_data[i] = self.convert_body_frame(imu, cosine_matrices[i])

        # remove time column from data
        if not include_time:
            imu_data = imu_data[:,1:]
            cosine_matrices = cosine_matrices[:,1:]

        if flatten_labels:
            if body_frame:
                return imu_data, activities_flat
            else:
                return imu_data, cosine_matrices, activities_flat
        else:
            if body_frame:
                return imu_data, activities, activities_bounds
            else:
                return imu_data, cosine_matrices, activities, activities_bounds

    def convert_body_frame(self, imu_data, cosine_matrix):
        """Converts the sensor frames in a single data item to body frames through the cosine matrix."""
        C = cosine_matrix[1:].reshape(3,3).T

        imu_data[1:4]  = np.dot(C, imu_data[1:4].T)  # acc
        imu_data[4:7]  = np.dot(C, imu_data[4:7].T)  # gyro
        imu_data[7:10] = np.dot(C, imu_data[7:10].T) # mag

        return imu_data

    def _reply_labels(self, labels, bounds):
        """Builds a single labels array from labels and bounds."""
        start = bounds[0::2]-1 # even positions
        stop  = bounds[1::2]   # -1: numbering starts from 1

        res = np.zeros(bounds[-1], dtype=np.uint8)
        for i, act in enumerate(labels):
            res[start[i] : stop[i]] = act
            # start is included, stop is excluded
        return res

    def _create_label_maps(self):
        """Builds labels mappings."""
        if hasattr(self,'map_encode') and hasattr(self,'map_decode'): return

        self.labels = [
            'RUNNING', 'WALKING', 'JUMPING', 'STNDING', 'SITTING', 'XLYINGX', 'FALLING',
            'TRANSUP', # getting up from lower position
            'TRANSDW', # getting down from upper position
            'TRNSACC', # accelerating
            'TRNSDCC', # decelerating
            'TRANSIT'  # other transitions or irrelevant
        ]

        self.map_encode = { l : np.uint8(i) for i,l in enumerate(self.labels) } # str > int
        self.map_decode = { np.uint8(i) : l for i,l in enumerate(self.labels) } # int > str

    def get_label_encode_map(self):
        return self.map_encode.copy()

    def get_label_decode_map(self):
        return self.map_decode.copy()

if __name__ == "__main__":
    # just a test
    imu_data, labels = ARSDataset('dataset/ARS_DLR_DataSet_V2.mat').load_dataset()
    print('Loaded')
