import os
from dotenv import load_dotenv
from app import create_app
from app.config.default import config

# Load environment variables from .env file
load_dotenv()

# Get environment from FLASK_ENV or default to development
env = os.getenv('FLASK_ENV', 'development')

# Create app with environment-specific config
app = create_app(config[env])

if __name__ == "__main__":
    # Get port from environment or default to 5001
    port = int(os.getenv('PORT', 5001))
    
    # Run the app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=app.config['DEBUG']
    )