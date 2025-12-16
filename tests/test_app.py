import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import create_app
from config import Config

class TestConfig(Config):
    TESTING = True
    DATABASE_PATH = ':memory:'
    # Mock model path to avoid loading heavy weights during quick tests if possible
    # For now, we'll let it try to load or fail gracefully in logs

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'HemaVision', response.data)

    def test_history_route(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
