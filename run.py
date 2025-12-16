import os
import sys

# Add src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from med_image.app.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()
