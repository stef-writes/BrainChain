# ScriptChain App

A modular and flexible workflow builder with a drag-and-drop interface. ScriptChain allows users to create, customize, and manage complex workflows through an intuitive visual interface.

## Features

- Visual workflow builder with drag-and-drop functionality
- Modular component system
- Real-time workflow validation
- Interactive node connections
- Modern, responsive user interface

## Tech Stack

### Frontend
- React 18
- ReactFlow for workflow visualization
- Axios for API communication

### Backend
- Python
- Flask (REST API)

## Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` (if it exists)
   - Configure the required environment variables

5. Start the backend server:
   ```bash
   python run.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` (if it exists)
   - Configure the required environment variables

4. Start the development server:
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`

## Project Structure

```
scriptchain-app/
├── backend/
│   ├── app/           # Main application code
│   ├── requirements.txt
│   └── run.py        # Entry point
└── frontend/
    ├── src/          # React source code
    ├── public/       # Static assets
    └── package.json
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

[Add your license information here]

## Support

For support, please [add contact information or issue reporting guidelines]