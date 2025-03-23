# ScriptChain

ScriptChain is a powerful application that combines the capabilities of LangChain and OpenAI to create and manage script chains. It features a modern web interface built with React and a robust Flask backend.

## Project Structure

```
scriptchain-app/
├── backend/                 # Flask backend application
│   ├── app/                # Main application package
│   │   ├── config/        # Configuration files
│   │   ├── models.py      # Database models
│   │   ├── routes.py      # API endpoints
│   │   ├── scriptchain.py # Core ScriptChain functionality
│   │   ├── context_manager.py # Context management
│   │   ├── model_config.py    # Model configuration
│   │   ├── node_types.py      # Node type definitions
│   │   ├── prompt_templates.py # Prompt templates
│   │   └── utils.py          # Utility functions
│   ├── requirements.txt    # Python dependencies
│   ├── run.py             # Application entry point
│   └── .env.example       # Example environment variables
└── frontend/              # React frontend application
    ├── src/              # Source code
    ├── public/           # Static files
    ├── package.json      # Node.js dependencies
    └── .env              # Frontend environment variables
```

## Features

- **Script Chain Management**: Create and manage chains of scripts using LangChain
- **Interactive UI**: Modern React-based interface with ReactFlow for visual chain editing
- **Database Integration**: SQLAlchemy-based persistence for chains and configurations
- **API Integration**: OpenAI integration for advanced language model capabilities
- **Context Management**: Sophisticated context handling for script execution
- **Error Handling**: Comprehensive error handling and logging
- **Health Monitoring**: Built-in health check endpoints

## Prerequisites

- Python 3.8+
- Node.js 14+
- OpenAI API key
- SQLite (or other supported database)

## Backend Setup

1. Create and activate a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
python run.py
```

The backend will start on port 5001 by default.

## Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the development server:
```bash
npm start
```

The frontend will start on port 3000 by default.

## API Endpoints

- `GET /api/health`: Health check endpoint
- `GET /api/chains`: List all script chains
- `POST /api/chains`: Create a new script chain
- `GET /api/chains/<id>`: Get a specific chain
- `PUT /api/chains/<id>`: Update a chain
- `DELETE /api/chains/<id>`: Delete a chain
- `POST /api/chains/<id>/execute`: Execute a chain

## Development

### Backend Development

- The backend uses Flask with SQLAlchemy for database operations
- LangChain is used for script chain management
- OpenAI integration for language model capabilities
- Comprehensive logging system for debugging

### Frontend Development

- React-based UI with ReactFlow for visual chain editing
- Axios for API communication
- Modern component-based architecture

## Testing

Backend tests can be run using pytest:
```bash
cd backend
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.