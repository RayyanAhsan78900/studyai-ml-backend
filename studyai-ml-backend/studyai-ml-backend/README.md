# StudyAI Pro — ML Backend

A tiny Flask API that plugs straight into StudyAI Pro's "AI / ML Backend"
settings panel. No numpy/scikit-learn — a small hand-written logistic
regression trained with plain-Python gradient descent, so it deploys as a
single lightweight function.

## Endpoints

- `GET /health` → `{status: "ok"}`
- `POST /predict/batch` → give it tasks, get back success probability + priority
- `POST /train` → give it your finished-task history, it retrains the model

These match exactly what the frontend already calls, so there's nothing to
change on the app side — just paste your deployed URL into
**Settings → AI / ML Backend → Backend URL**.

## Deploy option A — Vercel (recommended, easiest)

1. Push this folder to a new GitHub repo.
2. Go to [vercel.com/new](https://vercel.com/new), click **Import Project**,
   pick the repo. Vercel auto-detects `vercel.json` — no config needed.
3. Click **Deploy**. You'll get a URL like `https://your-app.vercel.app`.
4. Paste that URL into the app's Backend URL field and hit **Test Connection**.

⚠️ **Note on Vercel:** it's serverless — each deploy/cold start spins up a
fresh process, so anything learned via **Train on My Data** can reset after
periods of inactivity. This is fine for demoing the model, since the app
always falls back to its built-in heuristic if the backend hasn't been
trained. If you want training to *stick permanently*, use option B.

## Deploy option B — Render (persistent, keeps trained weights)

1. Push this folder to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect the
   repo. Render reads `render.yaml` automatically.
3. Deploy. You'll get a URL like `https://studyai-ml-backend.onrender.com`.
4. Same as above — paste it into the app's settings.

Render's free tier keeps one long-running process, so trained weights
persist between requests until the service redeploys or is manually restarted
(free tier services also spin down after inactivity and take ~30s to wake up
on the next request — that's normal).

## Local development

```bash
pip install -r requirements.txt
python api/index.py
# → running on http://localhost:5000
```

Then set Backend URL to `http://localhost:5000` while testing.

## How the model works (short version)

- Features per task: days left until deadline, hours needed, difficulty,
  your historical completion rate, and current streak.
- On first boot it bootstraps itself on synthetic data so predictions are
  sensible immediately.
- `/train` retrains it further on your real completed/overdue tasks —
  the more history you send, the more it reflects your actual habits.
- `priority_score` blends urgency, predicted risk of missing the deadline,
  and difficulty into a single 0–100 number, bucketed into `low` / `medium`
  / `high`.
