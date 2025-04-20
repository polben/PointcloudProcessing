import unittest

from UI import UI


class UIFunctionality(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_startUi(self):
        ui = UI(lauch=False)
        self.assertIsNone(None)