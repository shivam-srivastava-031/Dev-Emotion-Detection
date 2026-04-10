const API_BASE = "http://localhost:8000";

// ── Original endpoints ──

export async function analyzeText(text) {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Analysis failed");
  }
  return res.json();
}

export async function getTimeline(limit = 50) {
  const res = await fetch(`${API_BASE}/api/timeline?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch timeline");
  return res.json();
}

export async function getInsights() {
  const res = await fetch(`${API_BASE}/api/insights`);
  if (!res.ok) throw new Error("Failed to fetch insights");
  return res.json();
}

export async function getEmotions() {
  const res = await fetch(`${API_BASE}/api/emotions`);
  if (!res.ok) throw new Error("Failed to fetch emotions");
  return res.json();
}

// ── Behavior Engine endpoints ──

export async function getBehavior() {
  const res = await fetch(`${API_BASE}/api/behavior`);
  if (!res.ok) throw new Error("Failed to fetch behavior data");
  return res.json();
}

export async function getTransitions() {
  const res = await fetch(`${API_BASE}/api/behavior/transitions`);
  if (!res.ok) throw new Error("Failed to fetch transitions");
  return res.json();
}

export async function getTrends() {
  const res = await fetch(`${API_BASE}/api/behavior/trends`);
  if (!res.ok) throw new Error("Failed to fetch trends");
  return res.json();
}

// ── Dataset endpoints ──

export async function loadDataset(source, maxRows = 2000) {
  const res = await fetch(`${API_BASE}/api/datasets/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, max_rows: maxRows }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Load failed");
  }
  return res.json();
}

export async function getDatasetLoadStatus() {
  const res = await fetch(`${API_BASE}/api/datasets/load/status`);
  if (!res.ok) throw new Error("Failed to fetch load status");
  return res.json();
}

export async function getDatasetStats() {
  const res = await fetch(`${API_BASE}/api/datasets/stats`);
  if (!res.ok) throw new Error("Failed to fetch dataset stats");
  return res.json();
}

export async function exploreDataset({ source, emotion, page = 1, pageSize = 20 } = {}) {
  const params = new URLSearchParams();
  if (source) params.set("source", source);
  if (emotion) params.set("emotion", emotion);
  params.set("page", page);
  params.set("page_size", pageSize);
  const res = await fetch(`${API_BASE}/api/datasets/explore?${params}`);
  if (!res.ok) throw new Error("Failed to explore dataset");
  return res.json();
}
