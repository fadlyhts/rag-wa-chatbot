# Deploy to EasyPanel

## Option 1: Deploy from Git Repository (Recommended)

### 1. Push code to Git
```bash
cd D:\Magang\rag\backend
git init
git add .
git commit -m "Initial commit: WhatsApp RAG Chatbot Backend"

# Create GitHub repo and push
git remote add origin YOUR_REPO_URL
git push -u origin main
```

### 2. Create Service in EasyPanel
1. Go to your EasyPanel project
2. Click **"Create Service"**
3. Select **"From GitHub"**
4. Choose repository: `your-repo/whatsapp-rag-backend`
5. Service name: `chatbot-backend`
6. Port: `8000`

### 3. Configure Environment Variables
In EasyPanel service settings, add all variables from `.env.production`

### 4. Deploy
Click **"Deploy"** - EasyPanel will build Docker image and start service

### 5. Get Public URL
After deployment, you'll get URL like: `https://chatbot-backend.lrbevh.easypanel.host`

---

## Option 2: Deploy from Docker Image

### 1. Build Docker image locally
```bash
cd D:\Magang\rag\backend
docker build -t whatsapp-rag-backend .
```

### 2. Push to Docker Hub
```bash
docker tag whatsapp-rag-backend YOUR_USERNAME/whatsapp-rag-backend:latest
docker push YOUR_USERNAME/whatsapp-rag-backend:latest
```

### 3. Deploy in EasyPanel
1. Create Service → From Docker Image
2. Image: `YOUR_USERNAME/whatsapp-rag-backend:latest`
3. Port: `8000`
4. Add environment variables
5. Deploy

---

## Option 3: Manual Deploy (Easiest - No Git Needed)

### 1. Create MySQL & Redis in EasyPanel
1. Create **MySQL** service
   - Get connection URL
   - Create database: `whatsapp_chatbot`
   
2. Create **Redis** service (or use existing)
   - Get connection URL

### 2. Create App Service
1. Create **"App"** service in EasyPanel
2. Choose **Python** template
3. Upload your backend folder (zip it first)

### 3. Configure
- Port: `8000`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Add environment variables

---

## After Deployment

### Update WAHA Webhook URL
Once deployed, get your backend URL (e.g., `https://chatbot-backend.lrbevh.easypanel.host`)

Then update WAHA environment variables:
```env
WHATSAPP_HOOK_URL=https://chatbot-backend.lrbevh.easypanel.host/api/webhook
WHATSAPP_HOOK_EVENTS=message,message.any
```

Restart WAHA and you're done! 🎉

---

## Verify Deployment

### Test endpoints:
```bash
curl https://chatbot-backend.lrbevh.easypanel.host/health
curl https://chatbot-backend.lrbevh.easypanel.host/api/info
```

### Test webhook:
```bash
curl -X POST https://chatbot-backend.lrbevh.easypanel.host/api/webhook \
  -H "Content-Type: application/json" \
  -d '{"event":"message","payload":{"from":"test","body":"test"}}'
```

---

## Troubleshooting

### Check logs
In EasyPanel: Service → Logs tab

### Check environment variables
Make sure all required variables are set

### Database connection
Make sure MySQL service is running and reachable

### WAHA webhook
Make sure WAHA environment variables are updated and service is restarted
