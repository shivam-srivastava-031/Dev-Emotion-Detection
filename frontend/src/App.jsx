import { useState, useEffect, useCallback, useRef } from "react";
import "./index.css";
import {
  analyzeText, getTimeline, getInsights, getEmotions,
  getBehavior,
  loadDataset, getDatasetLoadStatus, getDatasetStats, exploreDataset,
} from "./api";

import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  ArcElement, BarElement,
  Title, Tooltip, Legend, Filler,
} from "chart.js";
import { Line, Pie, Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  ArcElement, BarElement,
  Title, Tooltip, Legend, Filler,
);

/* ── Emotion helpers ──────────────────────────── */
const EMOTION_EMOJIS = {
  anger: "😠", disgust: "🤢", fear: "😨", joy: "😊",
  neutral: "😐", sadness: "😢", surprise: "😲",
};

const INSIGHT_ICONS = {
  dominant_emotion: "🏆", emotion_streak: "🔥",
  emotional_volatility: "🌊", emotion_distribution: "📊",
  recent_shift: "🔄",
};

const DEFAULT_COLORS = {
  anger: "#ef4444", disgust: "#a855f7", fear: "#6366f1",
  joy: "#facc15", neutral: "#94a3b8", sadness: "#3b82f6", surprise: "#f97316",
};

function getEmotionColor(emotion, colors) {
  return colors[emotion] || "#94a3b8";
}

/* ── Tab Navigation ───────────────────────────── */
function TabNav({ tabs, active, onSelect }) {
  return (
    <nav className="tab-nav" id="tab-nav">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`tab-btn ${active === t.id ? "active" : ""}`}
          onClick={() => onSelect(t.id)}
          id={`tab-${t.id}`}
        >
          {t.icon} {t.label}
        </button>
      ))}
    </nav>
  );
}

/* ═══════════════════════════════════════════════
   TAB 1: ANALYZE (original functionality)
   ═══════════════════════════════════════════════ */

function InputSection({ onAnalyze, loading }) {
  const [text, setText] = useState("");
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    onAnalyze(text);
    setText("");
  };
  return (
    <section className="glass-card input-section" id="input-section">
      <h2>📝 Analyze Text</h2>
      <form className="input-area" onSubmit={handleSubmit}>
        <textarea id="text-input" value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste any text, chat message, or journal entry…" maxLength={5000} />
        <button type="submit" className="btn-analyze" disabled={loading || !text.trim()} id="analyze-btn">
          {loading ? (<><span className="spinner" /> Analyzing…</>) : (<>🔍 Analyze Emotion</>)}
        </button>
      </form>
    </section>
  );
}

function ResultSection({ result, colors }) {
  if (!result) {
    return (
      <section className="glass-card result-section" id="result-section">
        <h2>🎯 Detection Result</h2>
        <p className="result-empty">Submit text to see the emotion analysis</p>
      </section>
    );
  }
  const bgColor = getEmotionColor(result.emotion, colors);
  return (
    <section className="glass-card result-section" id="result-section">
      <h2>🎯 Detection Result</h2>
      <div className="result-display">
        <span className="result-emotion-badge" style={{ background: bgColor }}>
          {EMOTION_EMOJIS[result.emotion] || "❓"} {result.emotion}
        </span>
        <p className="result-confidence">
          Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
        </p>
        {result.all_scores && (
          <div className="score-bars">
            {Object.entries(result.all_scores).sort(([, a], [, b]) => b - a).map(([label, score]) => (
              <div className="score-bar-row" key={label}>
                <span className="score-bar-label">{label}</span>
                <div className="score-bar-track">
                  <div className="score-bar-fill"
                    style={{ width: `${score * 100}%`, background: getEmotionColor(label, colors) }} />
                </div>
                <span className="score-bar-value">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function TimelineSection({ timeline, colors }) {
  return (
    <section className="glass-card timeline-section" id="timeline-section">
      <h2>🕒 Emotion Timeline</h2>
      {timeline.length === 0 ? (
        <p className="timeline-empty">No entries yet. Start analyzing text!</p>
      ) : (
        <div className="timeline-list">
          {timeline.map((item) => {
            const color = getEmotionColor(item.emotion, colors);
            return (
              <div className="timeline-item" key={item.id} style={{ borderLeftColor: color }}>
                <span className="timeline-dot" style={{ background: color }} />
                <div className="timeline-content">
                  <p className="timeline-text" title={item.text}>{item.text}</p>
                  <div className="timeline-meta">
                    <span className="timeline-emotion-tag" style={{ background: color }}>
                      {EMOTION_EMOJIS[item.emotion]} {item.emotion}
                    </span>
                    <span>{new Date(item.created_at).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function InsightsSection({ insights, colors }) {
  return (
    <section className="glass-card insights-section full-width" id="insights-section">
      <h2>💡 Insights & Patterns</h2>
      {insights.length === 0 ? (
        <p className="insights-empty">Insights will appear after you've analyzed a few entries.</p>
      ) : (
        <div className="insights-grid">
          {insights.map((insight, idx) => (
            <div className="insight-card" key={idx}>
              <div className="insight-card-icon">{INSIGHT_ICONS[insight.type] || "📌"}</div>
              <h3>{insight.title}</h3>
              <p dangerouslySetInnerHTML={{ __html: (insight.description || "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
              {insight.type === "emotion_distribution" && insight.distribution && (
                <div className="distribution-bars">
                  {Object.entries(insight.distribution).map(([label, pct]) => (
                    <div className="distribution-row" key={label}>
                      <span className="distribution-label">{label}</span>
                      <div className="distribution-track">
                        <div className="distribution-fill" style={{ width: `${pct}%`, background: getEmotionColor(label, colors) }} />
                      </div>
                      <span className="distribution-value">{pct}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/* ═══════════════════════════════════════════════
   TAB 2: BEHAVIOR ENGINE (Charts + Patterns)
   ═══════════════════════════════════════════════ */

function BehaviorTab({ colors }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getBehavior().then(setData).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="glass-card section-pad"><p className="result-empty"><span className="spinner" /> Loading behavior data…</p></div>;
  if (!data) return <div className="glass-card section-pad"><p className="result-empty">No behavior data available.</p></div>;

  const trendPoints = data.trend?.data_points || [];
  const movingAvg = data.trend?.moving_average || [];

  // ── Emotion Trend Line Chart ──
  const trendChartData = {
    labels: trendPoints.map((_, i) => i + 1),
    datasets: [
      {
        label: "Valence",
        data: trendPoints.map((p) => p.valence),
        borderColor: "rgba(59, 130, 246, 0.5)",
        backgroundColor: "rgba(59, 130, 246, 0.05)",
        pointBackgroundColor: trendPoints.map((p) => getEmotionColor(p.emotion, colors)),
        pointRadius: 5,
        fill: true,
        tension: 0.3,
      },
      {
        label: "Moving Avg",
        data: movingAvg,
        borderColor: "#8b5cf6",
        borderDash: [6, 3],
        pointRadius: 0,
        borderWidth: 2,
        fill: false,
        tension: 0.4,
      },
    ],
  };

  const trendChartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } },
      title: { display: false },
    },
    scales: {
      x: { title: { display: true, text: "Entry #", color: "#64748b" }, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { title: { display: true, text: "Valence (0=negative, 1=positive)", color: "#64748b" }, min: 0, max: 1, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
    },
  };

  // ── Pie chart: emotion distribution from context ──
  const ctxEmotions = data.context?.emotions || [];
  const ctxCounts = {};
  ctxEmotions.forEach((e) => (ctxCounts[e] = (ctxCounts[e] || 0) + 1));
  const pieLabels = Object.keys(ctxCounts);
  const pieData = {
    labels: pieLabels,
    datasets: [{
      data: Object.values(ctxCounts),
      backgroundColor: pieLabels.map((l) => getEmotionColor(l, colors)),
      borderColor: "rgba(0,0,0,0.3)",
      borderWidth: 1,
    }],
  };

  // ── Transition Heatmap (as bar chart) ──
  const transitions = data.transitions?.transitions || [];
  const topTrans = transitions.slice(0, 10);

  const transBarData = {
    labels: topTrans.map((t) => `${t.from} → ${t.to}`),
    datasets: [{
      label: "Probability",
      data: topTrans.map((t) => t.probability),
      backgroundColor: topTrans.map((t) => getEmotionColor(t.from, colors) + "aa"),
      borderColor: topTrans.map((t) => getEmotionColor(t.from, colors)),
      borderWidth: 1,
    }],
  };
  const transBarOpts = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { min: 0, max: 1, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { ticks: { color: "#94a3b8", font: { family: "Inter", size: 11 } }, grid: { display: false } },
    },
  };

  // ── Behavior insights ──
  const behaviorInsights = data.insights || [];

  return (
    <div className="behavior-grid">
      {/* Trend Chart */}
      <section className="glass-card section-pad full-width" id="trend-chart-section">
        <h2>📈 Emotional Trend</h2>
        <p className="section-subtitle" dangerouslySetInnerHTML={{
          __html: (data.trend?.description || "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        }} />
        <div className="chart-container" style={{ height: 280 }}>
          {trendPoints.length > 0 ? <Line data={trendChartData} options={trendChartOpts} /> : <p className="result-empty">Not enough data for trend chart.</p>}
        </div>
      </section>

      {/* Pie + Transitions side by side */}
      <section className="glass-card section-pad" id="pie-chart-section">
        <h2>🥧 Emotion Distribution</h2>
        <div className="chart-container" style={{ height: 240 }}>
          {pieLabels.length > 0 ? <Pie data={pieData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", padding: 12, font: { family: "Inter" } } } } }} /> : <p className="result-empty">No data.</p>}
        </div>
      </section>

      <section className="glass-card section-pad" id="transitions-section">
        <h2>🔀 Emotion Transitions</h2>
        <div className="chart-container" style={{ height: 240 }}>
          {topTrans.length > 0 ? <Bar data={transBarData} options={transBarOpts} /> : <p className="result-empty">Not enough transitions.</p>}
        </div>
      </section>

      {/* Behavior Insights */}
      {behaviorInsights.length > 0 && (
        <section className="glass-card section-pad full-width" id="behavior-insights-section">
          <h2>🧠 Behavior Insights</h2>
          <div className="insights-grid">
            {behaviorInsights.map((ins, idx) => (
              <div className={`insight-card severity-${ins.severity || "low"}`} key={idx}>
                <div className="insight-card-icon">{ins.icon || "📌"}</div>
                <h3>{ins.title}</h3>
                <p dangerouslySetInnerHTML={{ __html: (ins.description || "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Loops */}
      {(data.transitions?.loops || []).length > 0 && (
        <section className="glass-card section-pad full-width" id="loops-section">
          <h2>🔄 Detected Emotional Loops</h2>
          <div className="loops-list">
            {data.transitions.loops.map((loop, idx) => (
              <div className="loop-item" key={idx}>
                <span className="loop-pattern">{loop.pattern.join(" → ")}</span>
                <span className="loop-count">×{loop.count}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════
   TAB 3: DATASET EXPLORER
   ═══════════════════════════════════════════════ */

function DatasetTab({ colors }) {
  const [loadingDS, setLoadingDS] = useState(false);
  const [loadStatus, setLoadStatus] = useState(null);
  const [stats, setStats] = useState(null);
  const [explore, setExplore] = useState(null);
  const [filters, setFilters] = useState({ source: "", emotion: "", page: 1 });
  const pollRef = useRef(null);

  const refreshStats = useCallback(() => {
    getDatasetStats().then(setStats).catch(console.error);
  }, []);

  useEffect(() => { refreshStats(); }, [refreshStats]);

  const handleLoad = async (source) => {
    setLoadingDS(true);
    try {
      await loadDataset(source, 2000);
      // Start polling
      pollRef.current = setInterval(async () => {
        const st = await getDatasetLoadStatus();
        setLoadStatus(st);
        if (st.status !== "loading") {
          clearInterval(pollRef.current);
          setLoadingDS(false);
          refreshStats();
        }
      }, 1500);
    } catch (e) {
      alert("Error: " + e.message);
      setLoadingDS(false);
    }
  };

  const handleExplore = async () => {
    const data = await exploreDataset(filters);
    setExplore(data);
  };

  useEffect(() => {
    if (filters.page) handleExplore();
  }, [filters.page]);

  return (
    <div className="dataset-grid">
      {/* Loader */}
      <section className="glass-card section-pad full-width" id="dataset-loader">
        <h2>📦 Load Datasets</h2>
        <div className="dataset-loader-row">
          <button className="btn-dataset" onClick={() => handleLoad("goemotions")} disabled={loadingDS} id="load-goemotions-btn">
            {loadingDS && loadStatus?.source === "goemotions" ? <><span className="spinner" /> Loading…</> : "Load GoEmotions (2K samples)"}
          </button>
          <button className="btn-dataset" onClick={() => handleLoad("meld")} disabled={loadingDS} id="load-meld-btn">
            {loadingDS && loadStatus?.source === "meld" ? <><span className="spinner" /> Loading…</> : "Load MELD / EmotionLines (1K)"}
          </button>
        </div>
        {loadStatus && loadStatus.status === "loading" && (
          <div className="progress-bar-container">
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${(loadStatus.processed / loadStatus.total) * 100}%` }} />
            </div>
            <span className="progress-text">{loadStatus.processed} / {loadStatus.total} processed</span>
          </div>
        )}
        {loadStatus && loadStatus.status === "done" && (
          <p className="load-success">✅ Loaded {loadStatus.processed} records from <strong>{loadStatus.source}</strong></p>
        )}
        {loadStatus && loadStatus.status === "error" && (
          <p className="load-error">❌ Error: {loadStatus.error}</p>
        )}
      </section>

      {/* Stats */}
      {stats && Object.keys(stats).length > 0 && (
        <section className="glass-card section-pad full-width" id="dataset-stats">
          <h2>📊 Dataset Statistics</h2>
          <div className="stats-grid">
            {Object.entries(stats).map(([source, s]) => (
              <div className="stat-card" key={source}>
                <h3>{source.toUpperCase()}</h3>
                <div className="stat-row"><span>Total Records</span><strong>{s.total}</strong></div>
                <div className="stat-row"><span>Model Accuracy</span>
                  <strong className={`accuracy-badge ${s.accuracy > 60 ? "good" : s.accuracy > 40 ? "ok" : "low"}`}>
                    {s.accuracy}%
                  </strong>
                </div>
                <div className="distribution-bars" style={{ marginTop: "0.75rem" }}>
                  {Object.entries(s.predicted_distribution || {}).sort(([,a],[,b]) => b - a).map(([emo, count]) => (
                    <div className="distribution-row" key={emo}>
                      <span className="distribution-label">{emo}</span>
                      <div className="distribution-track">
                        <div className="distribution-fill" style={{ width: `${(count / s.total) * 100}%`, background: getEmotionColor(emo, colors) }} />
                      </div>
                      <span className="distribution-value">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Explorer */}
      <section className="glass-card section-pad full-width" id="dataset-explorer">
        <h2>🔍 Browse Records</h2>
        <div className="explorer-filters">
          <select value={filters.source} onChange={(e) => setFilters({ ...filters, source: e.target.value, page: 1 })} id="filter-source">
            <option value="">All Sources</option>
            <option value="goemotions">GoEmotions</option>
            <option value="meld">MELD</option>
          </select>
          <select value={filters.emotion} onChange={(e) => setFilters({ ...filters, emotion: e.target.value, page: 1 })} id="filter-emotion">
            <option value="">All Emotions</option>
            {Object.keys(DEFAULT_COLORS).map((e) => <option key={e} value={e}>{e}</option>)}
          </select>
          <button className="btn-explore" onClick={() => { setFilters({ ...filters, page: 1 }); handleExplore(); }} id="explore-btn">Search</button>
        </div>

        {explore && (
          <>
            <p className="explore-count">{explore.total} records found — page {explore.page}/{explore.pages}</p>
            <div className="explore-table-wrap">
              <table className="explore-table" id="explore-table">
                <thead>
                  <tr><th>Text</th><th>Source</th><th>Ground Truth</th><th>Predicted</th><th>Conf.</th></tr>
                </thead>
                <tbody>
                  {explore.records.map((r) => (
                    <tr key={r.id}>
                      <td className="explore-text" title={r.text}>{r.text}</td>
                      <td><span className="source-badge">{r.source}</span></td>
                      <td><span className="emo-tag" style={{ background: getEmotionColor(r.mapped_emotion, colors) }}>{r.mapped_emotion}</span></td>
                      <td><span className="emo-tag" style={{ background: getEmotionColor(r.predicted_emotion, colors) }}>{r.predicted_emotion}</span></td>
                      <td>{(r.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <button disabled={explore.page <= 1} onClick={() => setFilters({ ...filters, page: filters.page - 1 })}>← Prev</button>
              <span>Page {explore.page} / {explore.pages}</span>
              <button disabled={explore.page >= explore.pages} onClick={() => setFilters({ ...filters, page: filters.page + 1 })}>Next →</button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   APP ROOT
   ═══════════════════════════════════════════════ */

const TABS = [
  { id: "analyze", icon: "📝", label: "Analyze" },
  { id: "behavior", icon: "🧠", label: "Behavior Engine" },
  { id: "datasets", icon: "📦", label: "Datasets" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [colors, setColors] = useState(DEFAULT_COLORS);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [insights, setInsights] = useState([]);

  const refreshData = useCallback(async () => {
    try {
      const [tl, ins] = await Promise.all([getTimeline(), getInsights()]);
      setTimeline(tl);
      setInsights(ins);
    } catch (e) {
      console.error("Error refreshing data:", e);
    }
  }, []);

  useEffect(() => {
    getEmotions().then(setColors).catch(() => setColors(DEFAULT_COLORS));
    refreshData();
  }, [refreshData]);

  const handleAnalyze = async (text) => {
    setLoading(true);
    try {
      const res = await analyzeText(text);
      setResult(res);
      await refreshData();
    } catch (e) {
      console.error("Analysis error:", e);
      alert("Analysis failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Emotion Detection Pipeline</h1>
        <p>Real-time emotion analysis powered by BERT — track your emotional journey</p>
      </header>

      <TabNav tabs={TABS} active={activeTab} onSelect={setActiveTab} />

      {activeTab === "analyze" && (
        <div className="main-grid">
          <InputSection onAnalyze={handleAnalyze} loading={loading} />
          <ResultSection result={result} colors={colors} />
          <TimelineSection timeline={timeline} colors={colors} />
          <InsightsSection insights={insights} colors={colors} />
        </div>
      )}

      {activeTab === "behavior" && <BehaviorTab colors={colors} />}
      {activeTab === "datasets" && <DatasetTab colors={colors} />}
    </div>
  );
}
