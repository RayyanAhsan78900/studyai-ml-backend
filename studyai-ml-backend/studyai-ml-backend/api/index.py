"""
StudyAI Pro — ML Backend
A tiny, dependency-light Flask API that matches the contract the
StudyAI Pro frontend already expects:

  GET  /health
  POST /predict/batch   { tasks: [ {id, days_left, hours, difficulty, completion_rate, streak}, ... ] }
  POST /train           { history: [ {days_left_at_creation, hours, difficulty, completion_rate, streak, completed_on_time}, ... ] }

No numpy / scikit-learn required — a small hand-rolled logistic
regression trained with plain-Python gradient descent. This keeps the
deployed function tiny and fast on serverless platforms (Vercel) while
still being a "real" trained model, not just a fixed formula.

NOTE ON STATE: serverless functions (Vercel) don't guarantee the same
process stays warm between requests, so anything trained via /train
can reset on a cold start or redeploy. That's fine here because the
frontend always falls back to its local heuristic if the backend is
unreachable or hasn't been trained recently. If you want training to
persist permanently, deploy this same Flask app to Render/Railway/Fly
instead (see README) — those keep one long-running process.
"""

import math
import random
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the frontend (any origin) to call this API

FEATURES = ["days_left", "hours", "difficulty", "completion_rate", "streak"]


# ─── Tiny logistic regression (pure Python, no deps) ────────────────────────
class TinyLogisticModel:
    def __init__(self):
        self.weights = [0.0] * len(FEATURES)
        self.bias = 0.0
        self.samples_seen = 0
        self._bootstrap()

    @staticmethod
    def _normalize(feats):
        days_left, hours, difficulty, completion_rate, streak = feats
        return [
            max(-1.0, min(1.0, days_left / 14.0)),
            max(-1.0, min(1.0, hours / 10.0)),
            (difficulty - 2.0) / 1.0,
            (completion_rate - 50.0) / 50.0,
            max(-1.0, min(1.0, streak / 14.0)),
        ]

    @staticmethod
    def _sigmoid(z):
        if z < -60:
            return 0.0
        if z > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, feats):
        x = self._normalize(feats)
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return self._sigmoid(z)

    def _train(self, data, epochs=150, lr=0.3):
        n = len(data)
        if n == 0:
            return
        for _ in range(epochs):
            grad_w = [0.0] * len(FEATURES)
            grad_b = 0.0
            for feats, label in data:
                x = self._normalize(feats)
                z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
                pred = self._sigmoid(z)
                err = pred - label
                for i in range(len(x)):
                    grad_w[i] += err * x[i]
                grad_b += err
            for i in range(len(self.weights)):
                self.weights[i] -= lr * grad_w[i] / n
            self.bias -= lr * grad_b / n
        self.samples_seen += n

    def _bootstrap(self):
        """Seed the model with synthetic-but-sensible training data so
        predictions are reasonable from the very first request, before
        any real user history has been sent to /train."""
        random.seed(42)
        data = []
        for _ in range(250):
            days_left = random.uniform(-3, 21)
            hours = random.uniform(0.5, 10)
            difficulty = random.choice([1, 2, 3])
            completion_rate = random.uniform(0, 100)
            streak = random.uniform(0, 20)
            # A student is more likely to finish on time when there's
            # more runway, fewer hours needed, lower difficulty, a
            # better historical completion rate, and a longer streak.
            score = (
                days_left * 2
                - hours * 3
                - difficulty * 8
                + completion_rate * 0.4
                + streak * 2
            )
            prob = 1 / (1 + math.exp(-score / 20))
            label = 1 if random.random() < prob else 0
            data.append(([days_left, hours, difficulty, completion_rate, streak], label))
        self._train(data, epochs=300, lr=0.5)


model = TinyLogisticModel()


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return jsonify(name="StudyAI Pro ML Backend", status="ok")


@app.route("/health")
def health():
    return jsonify(status="ok", samples_seen=model.samples_seen, time=time.time())


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(force=True, silent=True) or {}
    tasks = payload.get("tasks", [])
    predictions = []

    for t in tasks:
        feats = [
            float(t.get("days_left", 7)),
            float(t.get("hours", 2)),
            float(t.get("difficulty", 2)),
            float(t.get("completion_rate", 50)),
            float(t.get("streak", 0)),
        ]
        success_probability = model.predict_proba(feats) * 100

        # Priority blends urgency (days left), predicted risk (inverse
        # of success probability), and task difficulty.
        urgency = max(0.0, 100.0 - feats[0] * 5)
        priority_score = round(
            0.5 * urgency + 0.3 * (100 - success_probability) + 0.2 * (feats[2] * 33.3),
            1,
        )
        priority_score = max(0.0, min(100.0, priority_score))

        if priority_score >= 66:
            band = "high"
        elif priority_score >= 33:
            band = "medium"
        else:
            band = "low"

        predictions.append(
            {
                "id": t.get("id"),
                "success_probability": round(success_probability, 1),
                "priority_score": priority_score,
                "priority_band": band,
            }
        )

    return jsonify(predictions=predictions)


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(force=True, silent=True) or {}
    history = payload.get("history", [])

    if len(history) < 5:
        return jsonify(
            trained=False,
            reason=f"Need at least 5 finished/overdue tasks to train (got {len(history)}).",
        )

    data = []
    for h in history:
        feats = [
            float(h.get("days_left_at_creation", 7)),
            float(h.get("hours", 2)),
            float(h.get("difficulty", 2)),
            float(h.get("completion_rate", 50)),
            float(h.get("streak", 0)),
        ]
        label = 1 if h.get("completed_on_time") else 0
        data.append((feats, label))

    model._train(data, epochs=150, lr=0.3)

    return jsonify(
        trained=True,
        rows_used_this_call=len(data),
        total_real_samples_seen=model.samples_seen,
    )


# Local dev entrypoint: `python api/index.py`
if __name__ == "__main__":
    app.run(debug=True, port=5000)
