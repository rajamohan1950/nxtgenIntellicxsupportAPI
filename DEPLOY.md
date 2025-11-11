# Deployment Guide

## Deploy to Render.com (Free Tier)

### Option 1: Using Render Dashboard

1. Go to [Render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `rajamohan1950/nxtgenIntellicxsupportAPI`
4. Configure:
   - **Name**: `nxtgen-intellicxsupport-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m uvicorn src.backend.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
5. Click "Create Web Service"
6. Wait for deployment (5-10 minutes)
7. Your app will be live at: `https://nxtgen-intellicxsupport-api.onrender.com`

### Option 2: Using render.yaml (Recommended)

1. Push the `render.yaml` file to GitHub
2. Go to Render Dashboard → "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and deploy

### Environment Variables

No environment variables required for basic deployment.

### Post-Deployment

1. Visit your deployed URL
2. The frontend will automatically use the same domain for API calls
3. Test the `/health` endpoint: `https://your-app.onrender.com/health`

## Alternative: Railway.app (Free Tier)

1. Go to [Railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway will auto-detect Python and deploy
5. Add environment variable: `PORT` (auto-set by Railway)

## Alternative: Fly.io (Free Tier)

1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Follow prompts
4. Deploy: `fly deploy`

## Notes

- Free tier services may spin down after inactivity (15-30 min)
- First request after spin-down may take 30-60 seconds
- Consider upgrading for production use

