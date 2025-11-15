# StrideMatch MotionLab V2

Biomechanical running analysis app with shoe recommendations.

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (RECOMMENDED)

1. **Create a GitHub repository** with these files:
   - `app_fixed.py` (rename to `app.py`)
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with your GitHub account

4. **Click "New app"** and select:
   - Repository: your-repo
   - Branch: main
   - Main file path: app.py

5. **Deploy!** Your app will be live at `https://your-app.streamlit.app`

### Option 2: Render

1. Create account at [render.com](https://render.com)

2. Create a new **Web Service**

3. Connect your GitHub repo

4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

5. Add environment variable:
   - `PYTHON_VERSION` = `3.10.0`

### Option 3: Railway

1. Create account at [railway.app](https://railway.app)

2. New Project > Deploy from GitHub repo

3. Add a `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

## 📱 iPad/Mobile Optimization

The app includes several optimizations for iPad:
- Responsive layout
- Touch-friendly buttons
- Video playback controls
- Minimal toolbar mode

## ⚠️ Common Issues on iPad

1. **Video upload size**: Keep videos under 50MB
2. **Processing time**: Analysis may take longer on cloud servers
3. **Memory limits**: Free tiers have memory limitations

## 📁 File Structure

```
your-repo/
├── app.py                    # Main application (rename from app_fixed.py)
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages (ffmpeg)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                 # This file
```

## 🔧 Google Sheet Setup

Make sure your Google Sheet is:
1. **Public** (Share > Anyone with the link can view)
2. Has these columns (case-insensitive):
   - brand
   - model
   - terrain
   - stability
   - cushioning
   - weight_g
   - drop_mm
   - stack_mm
   - link 1
   - link 2
   - image_url
   - notes

## 💡 Tips for Best Performance

1. **Video quality**: 720p is optimal (not 4K)
2. **Video length**: 5-10 seconds is ideal
3. **Camera angle**: Profile view (side) works best
4. **Lighting**: Good lighting improves pose detection
