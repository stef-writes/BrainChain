from flask import Flask
from flask_cors import CORS
from .routes import create_routes

def create_app():
    """Create and configure the Flask application.
    
    Returns:
        The configured Flask application instance
    """
    app = Flask(__name__)
    
    # Configure CORS to accept requests from frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000"],  # React dev server
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Register routes
    create_routes(app)
    
    @app.route("/health")
    def health_check():
        return {"status": "healthy"}
    
    return app