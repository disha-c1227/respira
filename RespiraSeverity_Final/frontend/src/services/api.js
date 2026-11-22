const API_BASE_URL = 'http://127.0.0.1:8000/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE_URL}/model-info`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function validateImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/validate-image`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Validation failed' }));
    throw new Error(err.detail?.message || err.detail || 'Validation error');
  }
  return res.json();
}

export async function runUnifiedPredict(file, threshold = 70.0) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/predict?threshold=${threshold}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    const msg = typeof err.detail === 'object' ? err.detail.message : err.detail;
    throw new Error(msg || 'Prediction error');
  }
  return res.json();
}

export async function fetchDiseaseMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics/disease`);
  if (!res.ok) throw new Error('Failed to load disease metrics');
  return res.json();
}

export async function fetchThresholdMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics/thresholds`);
  if (!res.ok) throw new Error('Failed to load threshold metrics');
  return res.json();
}

export async function fetchSeverityMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics/severity`);
  if (!res.ok) throw new Error('Failed to load severity metrics');
  return res.json();
}

export async function fetchClusteringMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics/clustering`);
  if (!res.ok) throw new Error('Failed to load clustering metrics');
  return res.json();
}

export async function generateReportPdf(payload) {
  const res = await fetch(`${API_BASE_URL}/report/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to generate PDF report');
  return res.blob();
}
