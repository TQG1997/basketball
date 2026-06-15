"""Shared DataFactory — singleton for data normalization and train/valid split.

This is the single source of truth used by both src/ (training) and ui/ (inference).
"""

import numpy as np


class DataFactory(object):
    """Singleton pattern — instantiated once with raw .npy data.

    On first construction with real_data, it:
      1. Normalizes x/y/z positions (z-score) and stores mean/std.
      2. Splits data 9:1 into train/valid (no shuffle before split).
      3. Extracts team A (ball + offence) and team B (defence) subsets.

    Subsequent instantiations return the existing instance so that
    normalization stats and basket constants are available everywhere.
    """

    __instance = None

    def __new__(cls,
                real_data=None,
                seq_data=None,
                features_=None,
                real_feat=None):
        if not DataFactory.__instance:
            DataFactory.__instance = object.__new__(cls)
        else:
            print("Instance Exists! :D")
        return DataFactory.__instance

    def __init__(self,
                 real_data=None,
                 seq_data=None,
                 features_=None,
                 real_feat=None):
        """Initialize with dataset arrays.

        Parameters
        ----------
        real_data : np.ndarray, shape [N, length, 11, 4]
            entity 0=ball (x,y,z,flag), 1-5=offence (x,y,z,flag),
            6-10=defence (x,y,z,flag).
        seq_data : np.ndarray, shape [N, length, 12]
            Ball + 5 offence players' (x, y).
        features_ : np.ndarray, shape [N, length, 6]
            Ball-possession one-hot indicators.
        real_feat : np.ndarray, shape [N, length, 6]
            Same format as features_ for ground-truth plays.
        """
        if real_data is not None:
            self.__real_data = real_data
            self.__seq_data = seq_data
            self.features_ = features_
            self.real_feat = real_feat

            self.BASKET_LEFT = [4, 25]
            self.BASKET_RIGHT = [90, 25]

            # Normalize positions in-place on __real_data
            self.__norm_dict = self.__normalize_pos()

            # Prepare train/valid splits
            (self.train_data, self.valid_data,
             self.seq_train, self.seq_valid,
             self.f_train, self.f_valid,
             self.rf_train, self.rf_valid) = self.__get_ready()

    # ---------------------------------------------------------------
    #   Data accessors
    # ---------------------------------------------------------------

    def fetch_data(self):
        return self.train_data, self.valid_data

    def fetch_seq(self):
        return self.seq_train, self.seq_valid

    def fetch_feat(self):
        return self.f_train, self.f_valid

    def fetch_realF(self):
        return self.rf_train, self.rf_valid

    def fetch_ori_data(self):
        """Return original data as [N, seq_len, 23] (ball xyz + 10 players xy)."""
        return np.concatenate(
            [
                # ball xyz
                self.__real_data[:, :, 0, :3].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3]),
                # team A + team B players xy
                self.__real_data[:, :, 1:, :2].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 10 * 2])
            ],
            axis=-1)

    # ---------------------------------------------------------------
    #   Denormalization helpers
    # ---------------------------------------------------------------

    def recover_data(self, norm_data):
        """Denormalize x/y for a [..., 22] tensor (ball xy + 10 players xy)."""
        norm_data[:, :, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]] = (
            norm_data[:, :, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
            * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean'])
        norm_data[:, :, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = (
            norm_data[:, :, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
            * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean'])
        return norm_data

    def recover_play(self, norm_data):
        """Denormalize x/y for a [n_players, 22] tensor."""
        norm_data[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]] = (
            norm_data[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
            * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean'])
        norm_data[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = (
            norm_data[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
            * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean'])
        return norm_data

    def recover_seq(self, norm_data):
        """Denormalize x/y for a [..., 12] tensor (ball xy + 5 offence xy)."""
        norm_data[:, [0, 2, 4, 6, 8, 10]] = (
            norm_data[:, [0, 2, 4, 6, 8, 10]]
            * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean'])
        norm_data[:, [1, 3, 5, 7, 9, 11]] = (
            norm_data[:, [1, 3, 5, 7, 9, 11]]
            * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean'])
        return norm_data

    def recover_BALL_and_A(self, norm_data):
        """Denormalize ball + offence x/y for a [..., 12] tensor."""
        norm_data[:, :, [0, 2, 4, 6, 8, 10]] = (
            norm_data[:, :, [0, 2, 4, 6, 8, 10]]
            * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean'])
        norm_data[:, :, [1, 3, 5, 7, 9, 11]] = (
            norm_data[:, :, [1, 3, 5, 7, 9, 11]]
            * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean'])
        return norm_data

    def recover_B(self, norm_data):
        """Denormalize defence x/y for a [..., 22] tensor (indices 12-21)."""
        norm_data[:, :, [12, 14, 16, 18, 20]] = (
            norm_data[:, :, [12, 14, 16, 18, 20]]
            * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean'])
        norm_data[:, :, [13, 15, 17, 19, 21]] = (
            norm_data[:, :, [13, 15, 17, 19, 21]]
            * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean'])
        return norm_data

    # ---------------------------------------------------------------
    #   Shuffling
    # ---------------------------------------------------------------

    def shuffle_train(self):
        """Shuffle training data (called per epoch)."""
        shuffled = np.random.permutation(self.train_data['A'].shape[0])
        self.train_data['A'] = self.train_data['A'][shuffled]
        self.train_data['B'] = self.train_data['B'][shuffled]
        self.seq_train = self.seq_train[shuffled]
        self.f_train = self.f_train[shuffled]
        self.rf_train = self.rf_train[shuffled]

    def shuffle_valid(self):
        """Shuffle validation data."""
        shuffled = np.random.permutation(self.valid_data['A'].shape[0])
        self.valid_data['A'] = self.valid_data['A'][shuffled]
        self.valid_data['B'] = self.valid_data['B'][shuffled]
        self.seq_valid = self.seq_valid[shuffled]
        self.f_valid = self.f_valid[shuffled]
        self.rf_valid = self.rf_valid[shuffled]

    def shuffle(self):
        """Shuffle both training and validation data (convenience method)."""
        self.shuffle_train()
        self.shuffle_valid()
        return (self.train_data, self.valid_data,
                self.seq_train, self.seq_valid,
                self.f_train, self.f_valid,
                self.rf_train, self.rf_valid)

    # ---------------------------------------------------------------
    #   Normalization (for inference — normalize a single sketch)
    # ---------------------------------------------------------------

    def normalize(self, input_):
        """Normalize player x, y on input.

        Parameters
        ----------
        input_ : np.ndarray, shape [batch, seq_len, features]
            Features include ball xy and offence player xy.

        Returns
        -------
        np.ndarray with x/y columns z-normalized in-place.
        """
        input_[:, :, [0, 2, 4, 6, 8, 10]] = (
            (input_[:, :, [0, 2, 4, 6, 8, 10]] - self.__norm_dict['x']['mean'])
            / self.__norm_dict['x']['stddev'])
        input_[:, :, [1, 3, 5, 7, 9, 11]] = (
            (input_[:, :, [1, 3, 5, 7, 9, 11]] - self.__norm_dict['y']['mean'])
            / self.__norm_dict['y']['stddev'])
        return input_

    # ---------------------------------------------------------------
    #   Internal: train/valid split + position normalization
    # ---------------------------------------------------------------

    def __get_ready(self):
        train = {}
        valid = {}

        # Team A: ball xyz + 5 offence players xy
        team_A = np.concatenate(
            [
                self.__real_data[:, :, 0, :3].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3]),
                self.__real_data[:, :, 1:6, :2].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2])
            ],
            axis=-1)
        train['A'], valid['A'] = np.split(
            team_A, [self.__real_data.shape[0] // 10 * 9])

        s_train, s_valid = np.split(
            self.__seq_data, [self.__real_data.shape[0] // 10 * 9])

        print("Offence: ", train['A'].shape)
        print(valid['A'].shape)
        print("Simplified: ", s_train.shape)
        print(s_valid.shape)

        # Team B: 5 defence players xy
        team_B = self.__real_data[:, :, 6:11, :2].reshape(
            [self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2])
        train['B'], valid['B'] = np.split(
            team_B, [self.__real_data.shape[0] // 10 * 9])

        f_train, f_valid = np.split(
            self.features_, [self.features_.shape[0] // 10 * 9])

        rf_train, rf_valid = np.split(
            self.real_feat, [self.real_feat.shape[0] // 10 * 9])

        print(f_valid.shape)

        return (train, valid,
                s_train, s_valid,
                f_train, f_valid,
                rf_train, rf_valid)

    def __normalize_pos(self):
        """Z-normalize x, y, z positions in-place on __real_data.

        Also normalizes BASKET_LEFT and BASKET_RIGHT constants.
        """
        norm_dict = {}
        axis_list = ['x', 'y', 'z']

        for i, axis_ in enumerate(axis_list):
            if axis_ == 'z':  # z — ball-only
                mean_ = np.mean(self.__real_data[:, :, 0, i])
                stddev_ = np.std(self.__real_data[:, :, 0, i])
                self.__real_data[:, :, 0, i] = (
                    (self.__real_data[:, :, 0, i] - mean_) / stddev_)
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
            else:  # x and y — all entities
                mean_ = np.mean(self.__real_data[:, :, :, i])
                stddev_ = np.std(self.__real_data[:, :, :, i])
                self.__real_data[:, :, :, i] = (
                    (self.__real_data[:, :, :, i] - mean_) / stddev_)

                self.BASKET_LEFT[i] = (self.BASKET_LEFT[i] - mean_) / stddev_
                self.BASKET_RIGHT[i] = (self.BASKET_RIGHT[i] - mean_) / stddev_
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
        return norm_dict
