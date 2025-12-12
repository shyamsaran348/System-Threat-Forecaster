// frontend/src/App.jsx
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5001";

function App() {
  // Form state matching the top features and input requirements
  const [formData, setFormData] = useState({
    EngineVersion: "1.1.15100.1",
    OSVersion: "10.0.17134.1",
    TotalPhysicalRAMMB: 4096,
    DateOS: "2018-01-01",
    DateAS: "2018-01-01",
    IsSecureBootEnabled: false,
    FirewallEnabled: true,
    IsPassiveModeEnabled: false,
    IsGamer: false,
    HasTpm: true,
  });

  const [predResult, setPredResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPredResult(null);

    try {
      // Prepare payload
      // Ensure numeric values are numbers
      const payload = {
        instance: {
          ...formData,
          TotalPhysicalRAMMB: Number(formData.TotalPhysicalRAMMB),
          IsSecureBootEnabled: formData.IsSecureBootEnabled ? 1 : 0,
          FirewallEnabled: formData.FirewallEnabled ? 1 : 0,
          IsPassiveModeEnabled: formData.IsPassiveModeEnabled ? 1 : 0,
          IsGamer: formData.IsGamer ? 1 : 0,
          HasTpm: formData.HasTpm ? 1 : 0,
        },
      };

      const res = await axios.post(`${API_BASE}/api/predict`, payload);
      if (res.data.predictions && res.data.predictions.length > 0) {
        setPredResult(res.data.predictions[0]);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const openGlobalShap = () => {
    window.open(`${API_BASE}/api/shap/global`, "_blank");
  };

  return (
    <div className="container">
      <header>
        <h1>System Threat Forecaster</h1>
        <p>Real-time Malware Infection Probability Prediction</p>
      </header>

      <main className="main-content">
        <section className="card form-card">
          <h2>System Telemetry Input</h2>
          <form onSubmit={handlePredict}>
            <div className="form-group">
              <label>Engine Version</label>
              <input
                type="text"
                name="EngineVersion"
                value={formData.EngineVersion}
                onChange={handleChange}
                placeholder="e.g. 1.1.15100.1"
              />
            </div>
            <div className="form-group">
              <label>OS Version</label>
              <input
                type="text"
                name="OSVersion"
                value={formData.OSVersion}
                onChange={handleChange}
                placeholder="e.g. 10.0.17134.1"
              />
            </div>
            <div className="form-group">
              <label>RAM (MB)</label>
              <input
                type="number"
                name="TotalPhysicalRAMMB"
                value={formData.TotalPhysicalRAMMB}
                onChange={handleChange}
              />
            </div>
            <div className="form-group">
              <label>OS Date</label>
              <input
                type="date"
                name="DateOS"
                value={formData.DateOS}
                onChange={handleChange}
              />
            </div>
            <div className="form-group">
              <label>AS Date</label>
              <input
                type="date"
                name="DateAS"
                value={formData.DateAS}
                onChange={handleChange}
              />
            </div>

            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  name="IsSecureBootEnabled"
                  checked={formData.IsSecureBootEnabled}
                  onChange={handleChange}
                />
                Secure Boot Enabled
              </label>
              <label>
                <input
                  type="checkbox"
                  name="FirewallEnabled"
                  checked={formData.FirewallEnabled}
                  onChange={handleChange}
                />
                Firewall Enabled
              </label>
              <label>
                <input
                  type="checkbox"
                  name="IsPassiveModeEnabled"
                  checked={formData.IsPassiveModeEnabled}
                  onChange={handleChange}
                />
                Passive Mode
              </label>
              <label>
                <input
                  type="checkbox"
                  name="IsGamer"
                  checked={formData.IsGamer}
                  onChange={handleChange}
                />
                Is Gamer
              </label>
              <label>
                <input
                  type="checkbox"
                  name="HasTpm"
                  checked={formData.HasTpm}
                  onChange={handleChange}
                />
                Has TPM
              </label>
            </div>

            <button type="submit" disabled={loading} className="predict-btn">
              {loading ? "Analyzing..." : "Predict Threat Level"}
            </button>
          </form>
          {error && <div className="error-msg">{error}</div>}
        </section>

        <section className="card result-card">
          <h2>Prediction Result</h2>
          {predResult ? (
            <div className={`result-box ${predResult.risk.toLowerCase()}`}>
              <div className="probability">
                {(predResult.probability * 100).toFixed(1)}%
              </div>
              <div className="risk-label">{predResult.risk} Risk</div>
              <p className="recommendation">
                {predResult.risk === "High"
                  ? "Immediate action recommended. Run full scan."
                  : predResult.risk === "Medium"
                    ? "Monitor system closely. Update definitions."
                    : "System appears healthy."}
              </p>
            </div>
          ) : (
            <div className="placeholder-result">
              Submit form to see prediction.
            </div>
          )}

          <div className="shap-actions">
            <h3>Explainability</h3>
            <button onClick={openGlobalShap} className="secondary-btn">
              View Global SHAP Summary
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
