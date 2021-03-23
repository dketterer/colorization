import unittest

from colorization.train import fill_growing_parameters


class MyTestCase(unittest.TestCase):
    def test_fill_growing_parameters(self):
        complete = fill_growing_parameters({0: 0,
                                            2: 1}, 4)
        self.assertEqual({0: 0,
                          1: 0,
                          2: 1,
                          3: 1}, complete)

        with self.assertRaises(AssertionError):
            fill_growing_parameters({2: 1}, 4)


if __name__ == '__main__':
    unittest.main()
