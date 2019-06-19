# DEPRECATED in favor of preprocessing notebook
import numpy as np
import scipy.io

class ARSDataset():
    """Interface to the ARS dataset."""

    def __init__(self, dataset_paths=None):
        """
        Initializes the class.

        dataset_paths: single dataset (string type) or list of datasets (list type) to load.
        """

        self._create_label_maps()

        if dataset_paths:
            if type(dataset_paths) == str:
                self.load_dataset(dataset_paths)
            elif type(dataset_paths) == list:
                for path in dataset_paths:
                    self.load_dataset(path)

    def get(self):
        """Returns sensor data and labels."""
        return self.data_dict['imu'], self.data_dict['act']

    def load_dataset(self, path, body_frame=True, include_time=True):
        """Loads a dataset. Appends new items to old data, if present."""

        # skip processing if saved dataset is passed
        if 'npz' in path:
            print('Loading from npz file: {}'.format(path))
            self._load_npz(path)
            return self.data_dict['imu'], self.data_dict['act']

        # load dataset from matlab mat file
        print('\nLoading from mat file: {}'.format(path))
        dataset = scipy.io.loadmat(path)
        keys = [ i for i in sorted(dataset) if '__' not in i ]

        # create data dictionary if not present
        if not hasattr(self, 'data_dict'):
            self.data_dict = {
                'imu': np.empty((0,10 if include_time else 9)),
                'cos': np.empty((0,10 if include_time else 9)),
                'act': np.empty((0,1), dtype=np.uint8),
                'ref': 'body' if body_frame else 'sensor'
            }

        # collect data from all tests
        for test in keys:
            print('Loading {}:'.format(test).ljust(52,' '), end='')
            test_dict = self._get_single_test_data(dataset, test, body_frame, include_time)

            # append values to global data dictionary
            self.data_dict['imu'] = np.append(self.data_dict['imu'], test_dict['imu'], axis=0)
            self.data_dict['cos'] = np.append(self.data_dict['cos'], test_dict['cos'], axis=0)
            self.data_dict['act'] = np.append(self.data_dict['act'], test_dict['act'])
            print('{} elements'.format(test_dict['imu'].shape[0]).rjust(15,' '))

    def _load_npz(self, path):
        self.data_dict = np.load(path)

    def save_dataset(self, path):
        """Saves current dataset to npz file."""

        print('Saving to npz file: {}'.format(path))
        np.savez_compressed(path, **self.data_dict)

    def _get_single_test_data(self, dataset, test_key, body_frame=True, include_time=True):
        """
        Gets data of a single test from the dataset.

        dataset: dataset to use
        test_key: the key representing the test to process
        body_frame: whether to convert coordinates to body frame, instead of using plain sensor frame
        include_time: whether to include or remove time column from sensor data
        """

        imu_data, cosine_matrices, activities, activities_bounds = dataset[test_key][0]

        # throwing away useless nested arrays
        activities = np.array([ act[0] for act in activities[0] ])
        activities_bounds = activities_bounds[0]

        # integrity checks on time and length
        assert([ imu_data[i][0] == cosine_matrices[i][0] for i in range(len(imu_data)) ])
        assert(len(activities_bounds) == 2*len(activities))

        # change labels to int numbers
        activities = np.array([ self.map_encode[a] for a in activities ])

        # get a single array of labels instead of labels + bounds
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

        return {
            'imu': imu_data,
            'cos': cosine_matrices,
            'act': activities_flat
        }

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
        # start is included, stop is excluded

        res = np.zeros(bounds[-1], dtype=np.uint8)
        for i, act in enumerate(labels):
            res[start[i] : stop[i]] = act
        return res

    def _create_label_maps(self):
        """Builds labels mappings."""
        if hasattr(self,'map_encode') and hasattr(self,'map_decode'): return

        # three dataset can be handled:
        # - v2 and benchmark share the same labels
        # - v1 has some specific labels

        self.map_encode = {
            'RUNNING': 0,
            'WALKING': 1,
            'JUMPING': 2,
            'STNDING': 3,
            'SITTING': 4,
            'XLYINGX': 5,
            'FALLING': 6,

            'WALKUPS': 1, # walking up and downstairs, v1-only labels
            'WALKDWS': 1, # mapped to walking

            'JUMPVRT': 2, # jumping in place, forward and backward
            'JUMPFWD': 2, # mapped to jumping
            'JUMPBCK': 2,

            'TRANSUP': 7, # getting up from lower pos, down from upper pos,
            'TRANSDW': 7, # accelerating, decelerating,
            'TRNSACC': 7, # and other transitions all mapped to the same value
            'TRNSDCC': 7,
            'TRANSIT': 7
        }

        self.map_decode = {
            0: 'running',
            1: 'walking',
            2: 'jumping',
            3: 'standing',
            4: 'sitting',
            5: 'lying',
            6: 'falling',
            7: 'transition'
        }

    def get_label_encode_map(self):
        return self.map_encode.copy()

    def get_label_decode_map(self):
        return self.map_decode.copy()

if __name__ == "__main__":
    # load mat files and save a single npz file ready to use
    ds = ARSDataset([
        'ARS_DLR_DataSet.mat',
        'ARS_DLR_DataSet_V2.mat',
        'ARS_DLR_Benchmark_Data_Set.mat'
    ])
    ds.save_dataset('ARS_DLR.npz')
    print('Done.')
