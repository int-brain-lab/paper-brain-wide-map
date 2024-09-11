import numpy as np
import pickle
from pathlib import Path
import unittest

from one.api import ONE

from brainwidemap.single_cell_stats.single_cell_Working_example_block import BWM_block_test
from brainwidemap.single_cell_stats.single_cell_Working_example_stimulus import BWM_stim_test
from brainwidemap.single_cell_stats.single_cell_Working_example_choice import BWM_choice_test
from brainwidemap.single_cell_stats.single_cell_Working_example_feedback import BWM_feedback_test


class TestSingleCell(unittest.TestCase):
    data = {}

    def setUp(self) -> None:
        self.one = ONE(base_url='https://alyx.internationalbrainlab.org')
        np.random.seed(0)
        fixture = Path(__file__).with_name('single_cell_data.pkl')
        if fixture.exists():
            with open(fixture, 'rb') as fp:
                self.data = pickle.load(fp)

    def test_block_test(self):
        pid = '3675290c-8134-4598-b924-83edb7940269'
        eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
        self._verify('block', BWM_block_test(self.one, pid, eid))

    def test_stim_test(self):
        pid = '3675290c-8134-4598-b924-83edb7940269'
        eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
        self._verify('stim', BWM_stim_test(self.one, pid, eid))

    def test_feedback_test(self):
        # The example session
        pid = '56f2a378-78d2-4132-b3c8-8c1ba82be598'
        eid = '6713a4a7-faed-4df2-acab-ee4e63326f8d'
        self._verify('feedback', BWM_feedback_test(self.one, pid, eid))

    def test_choice_test(self):
        # The example session
        pid = '3675290c-8134-4598-b924-83edb7940269'
        eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
        self._verify('choice', BWM_choice_test(self.one, pid, eid))

    def _verify(self, test, output):
        if test not in self.data:
            print('Updating test data...')
            self._update_data(test, output)
            self._save()
        expected = self.data[test]
        self.assertTrue(len(expected.dtype) == len(output),
                        f'Expected {len(expected.dtype)} outputs, got {len(output)}')
        for val, var in zip(output, expected.dtype.names):
            with self.subTest(var):
                np.testing.assert_array_equal(val, expected[var])

    def _update_data(self, test, output):
        names = ['p_1', 'p_2', 'area_label', 'QC_cluster_id']
        if len(output) == 3:
            names.pop(1)
        data = np.rec.fromarrays(output, names=names)
        self.data[test] = data

    def _save(self):
        fixture = Path(__file__).with_name('single_cell_data.pkl')
        with open(fixture, 'wb') as fp:
            pickle.dump(self.data, fp)

    def tearDown(self) -> None:
        np.random.seed()


if __name__ == '__main__':
    unittest.main()
