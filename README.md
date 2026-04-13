# Zomato PM Interview Simulator

A web application that simulates Product Manager interviews at Zomato using AI-powered Root Cause Analysis.

## Features

- Interactive chat-based interview simulation
- Multiple case studies
- AI-powered interviewer using Google Generative AI (with fallback for quota limits)
- Session management with cookies

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: Create a `.env` file with `GOOGLE_API_KEY=your_google_api_key`
4. Run locally: `uvicorn main_app:app --reload`

## Important Notes

- **API Quota**: The free tier of Google Generative AI has limited requests. If you exceed the quota, the app will show helpful fallback responses based on common RCA practices.
- **Model**: Uses `gemini-pro-latest` model. If you have a paid API key, you can upgrade for higher limits.
- **Deployment**: For Vercel deployment, ensure your API key has sufficient quota.

## Deployment on Vercel

1. Push this code to a GitHub repository
2. Connect the repository to Vercel
3. Set the `GOOGLE_API_KEY` environment variable in Vercel's dashboard
4. Deploy

The `vercel.json` is configured for FastAPI deployment.

## Files

- `main_app.py`: FastAPI application
- `templates/index.html`: HTML template
- `data/`: Knowledge base files
- `requirements.txt`: Python dependencies
- `vercel.json`: Vercel configuration