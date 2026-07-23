Render deployment guide
======================

This repository includes a Docker-based web UI for Kronos under `webui/`. Use the steps below to deploy to Render (https://render.com) using the included Dockerfile and render.yaml manifest.

Quick summary
-------------
- The service runs a Flask app via Gunicorn on port `7070`.
- The repository contains `webui/Dockerfile` and `render.yaml` for Render.

Steps
-----

1. Push your code to GitHub (e.g. `master` or `main`).

```bash
git add .
git commit -m "Add Docker + Render manifest"
git push origin master
```

2. Create a Render account and connect your GitHub account.

3. In Render, create a new service and choose "Web Service".
   - Connect the repository `shiyu-coder/Kronos` (or your fork `Thrishok/House-price-prediction-` if you pushed the branch there).
   - Render will detect the `render.yaml` and create the `kronos-webui` service.
   - Confirm Docker environment and `webui/Dockerfile` is selected.

4. Configuration:
   - Branch: `master` (or the branch you pushed)
   - Health check path: `/`
   - Environment variables (optional):
     - `PYTHONUNBUFFERED=1`
     - `FLASK_ENV=production`

5. (Optional) Increase instance disk if you want to cache large model weights on the instance filesystem.
   - In Render service settings, set Disk to at least `1 GB` (or larger depending on model size).

6. Deploy: Render will build the Docker image and start the service automatically.

Notes & caveats
---------------
- First model load (`/api/load-model`) may download model weights; allow time and ensure instance has network access and disk space.
- This Dockerfile is CPU-only. For GPU inference you need a GPU-enabled plan and a CUDA-enabled base image; adjust Dockerfile and PyTorch wheel accordingly.
- If you prefer to use the fork you own, push the branch to your fork and connect that repo in Render instead.

Troubleshooting
---------------
- If builds fail due to memory/time limits, consider using smaller models (e.g. `kronos-mini`) or pre-building the image and pushing to Docker Hub, then use Render's "Private Docker" option.
- Check service logs in Render dashboard for runtime errors.
