from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        RotatingFileHandler(
            'app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app(config_class: Optional[type] = None) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config_class: Optional configuration class to use
        
    Returns:
        The configured Flask application instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        from .config.default import Config
        config_class = Config
    app.config.from_object(config_class)
    
    # Validate required environment variables
    required_env_vars = ['OPENAI_API_KEY', 'DATABASE_URL']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": app.config.get('CORS_ORIGINS', ["http://localhost:3000"]),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Configure SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///scriptchain.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Configure engine options based on database type
    database_url = app.config['SQLALCHEMY_DATABASE_URI']
    if database_url.startswith('sqlite'):
        # SQLite doesn't support connection pooling
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {}
    else:
        # For other databases, use connection pooling
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 10,
            'pool_recycle': 3600,
            'pool_pre_ping': True
        }
    
    # Initialize extensions
    db.init_app(app)
    
    # Create database tables
    with app.app_context():
        try:
            db.Model.metadata.create_all(db.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    # Import and register routes after app creation to avoid circular imports
    from .routes import create_routes
    create_routes(app)
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        logging.warning(f"404 error: {error}")
        return jsonify({"error": "Not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logging.error(f"500 error: {error}")
        return jsonify({"error": "Internal server error"}), 500
    
    @app.errorhandler(ValueError)
    def validation_error(error):
        logging.warning(f"Validation error: {error}")
        return jsonify({"error": str(error)}), 400
    
    # Health check endpoint
    @app.route("/health")
    def health_check():
        try:
            # Check database connection
            db.session.execute("SELECT 1")
            status = "healthy"
            message = "Application is running normally"
        except Exception as e:
            logging.error(f"Health check failed: {str(e)}")
            status = "unhealthy"
            message = str(e)
            
        return jsonify({
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    # Store start time
    app.config['START_TIME'] = datetime.now().isoformat()
    
    logger.info("Application initialized successfully")
    return app