from app import app
import unittest

class TestProductionRoutes(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Home', response.data)

    def test_about(self):
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About', response.data)

    def test_prediction_page(self):
        response = self.app.get('/prediction_page')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predict', response.data)

    def test_analyze_history(self):
        response = self.app.get('/analyze')
        self.assertEqual(response.status_code, 200)
        # Even if DB is empty, it should return 200 and 'History' title
        self.assertIn(b'History', response.data)

if __name__ == '__main__':
    unittest.main()
