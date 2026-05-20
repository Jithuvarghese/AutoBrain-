# 🤖 AutoBrain — No-Code ML Platform

A fully local, no-code machine learning platform. Upload a dataset, clean it,
engineer features, train a model, and evaluate it — all through a guided step-by-step UI.
No coding required. No internet needed.

## Prerequisites
- Python 3.10+
- Node.js 18+
- pip

## Quick Start

### 1. Start the Backend
```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend (new terminal)
```powershell
cd frontend
npm install
npm run dev
```

### 3. Open the app
http://localhost:5173

## Project Storage
All data stored in `backend/storage/projects/{project_id}/`
No database required. No internet connection required.

## Tech Stack
- Frontend: React 18 + Vite + Tailwind CSS + Recharts
- Backend: Python + FastAPI + scikit-learn + pandas