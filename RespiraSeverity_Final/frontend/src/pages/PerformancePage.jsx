import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  CheckCircle2, 
  ShieldCheck, 
  AlertCircle,
  Activity,
  Table as TableIcon,
  ShieldAlert
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { fetchDiseaseMetrics, fetchSeverityMetrics, fetchClusteringMetrics, fetchThresholdMetrics } from '../services/api';

export default function PerformancePage() {
  const [diseaseData, setDiseaseData] = useState(null);
  const [severityData, setSeverityData] = useState(null);
  const [clusteringData, setClusteringData] = useState(null);
  const [thresholdData, setThresholdData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [d, s, c, t] = await Promise.all([
          fetchDiseaseMetrics(),
          fetchSeverityMetrics(),
          fetchClusteringMetrics(),
          fetchThresholdMetrics()
        ]);
        setDiseaseData(d);
        setSeverityData(s);
        setClusteringData(c);
        setThresholdData(t);
      } catch (e) {
        console.error('Failed to load performance metrics:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div className="py-20 text-center text-xs text-slate-500">Loading evaluation benchmarks...</div>;
  }

  // Training curves
  const trainingCurves = (diseaseData?.training_history?.epochs || [1, 2, 3, 4, 5, 6, 7, 8]).map((ep, i) => ({
    epoch: `Epoch ${ep}`,
    trainAcc: diseaseData.training_history.train_acc[i],
    valAcc: diseaseData.training_history.val_acc[i],
    trainLoss: diseaseData.training_history.train_loss[i],
    valLoss: diseaseData.training_history.val_loss[i],
  }));

  // Threshold curve data
  const threshCurve = (thresholdData?.thresholds || []).map(item => ({
    threshold: `${Math.round(item.threshold * 100)}%`,
    coverage: item.coverage_percent,
    accuracy: item.selective_accuracy,
    macroF1: item.macro_f1,
    rejection: item.rejection_percent
  }));

  return (
    <div className="space-y-10 py-4">
      {/* Header */}
      <div>
        <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[11px] font-semibold mb-1">
          <BarChart3 className="w-3.5 h-3.5 text-teal-600" />
          <span>Research Evaluation Protocol</span>
        </div>
        <h2 className="text-2xl font-bold text-navy-900 tracking-tight">
          Comprehensive Model Performance Dashboard
        </h2>
        <p className="text-slate-500 text-xs mt-0.5">
          Stratified 80:20 holdout evaluation results, selective classification curves, and training convergence.
        </p>
      </div>

      {/* SECTION 1: FOUR-CLASS DISEASE CLASSIFIER */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-4 border-b border-slate-100">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-teal-700 bg-teal-50 px-2 py-0.5 rounded">
              Primary Backbone
            </span>
            <h3 className="text-lg font-bold text-navy-900 mt-1">
              Four-Class Disease Classifier Performance (Holdout Set)
            </h3>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block uppercase">Overall Accuracy</span>
              <span className="text-xl font-extrabold text-teal-700">{diseaseData?.accuracy}%</span>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block uppercase">Macro F1-Score</span>
              <span className="text-xl font-extrabold text-navy-900">{diseaseData?.macro_f1}%</span>
            </div>
          </div>
        </div>

        {/* Per-class table */}
        <div>
          <h4 className="text-xs font-bold text-navy-900 mb-3 flex items-center space-x-2">
            <TableIcon className="w-4 h-4 text-teal-700" />
            <span>Per-Class Classification Report (5,687 Holdout Radiographs)</span>
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                  <th className="py-2.5 px-4 rounded-l-lg">Disease Category</th>
                  <th className="py-2.5 px-4">Precision</th>
                  <th className="py-2.5 px-4">Recall</th>
                  <th className="py-2.5 px-4">F1-Score</th>
                  <th className="py-2.5 px-4 rounded-r-lg">Support (Images)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {Object.entries(diseaseData?.per_class || {}).map(([clsName, metrics]) => (
                  <tr key={clsName} className="hover:bg-slate-50/70">
                    <td className="py-2.5 px-4 font-bold text-navy-900">{clsName}</td>
                    <td className="py-2.5 px-4">{metrics.precision}%</td>
                    <td className="py-2.5 px-4">{metrics.recall}%</td>
                    <td className="py-2.5 px-4 font-semibold text-teal-800">{metrics.f1}%</td>
                    <td className="py-2.5 px-4 text-slate-500">{metrics.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrix Table */}
        <div>
          <h4 className="text-xs font-bold text-navy-900 mb-3">Confusion Matrix (Rows = Actual, Columns = Predicted)</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-center border border-slate-200 rounded-xl overflow-hidden">
              <thead className="bg-navy-900 text-white font-semibold">
                <tr>
                  <th className="py-2 px-3 text-left">Actual \ Predicted</th>
                  {diseaseData?.confusion_matrix?.labels.map(l => (
                    <th key={l} className="py-2 px-3">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-slate-800">
                {diseaseData?.confusion_matrix?.matrix.map((row, rIdx) => {
                  const actualName = diseaseData.confusion_matrix.labels[rIdx];
                  return (
                    <tr key={actualName} className="hover:bg-slate-50">
                      <td className="py-2.5 px-4 text-left font-bold bg-slate-50 text-navy-900 border-r border-slate-200">
                        {actualName}
                      </td>
                      {row.map((val, cIdx) => (
                        <td 
                          key={cIdx} 
                          className={`py-2 px-3 font-semibold ${
                            rIdx === cIdx ? 'bg-teal-50 text-teal-900 font-bold' : val > 0 ? 'text-amber-800' : 'text-slate-300'
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Training Curves */}
        <div>
          <h4 className="text-xs font-bold text-navy-900 mb-3">Training & Validation Convergence Curves</h4>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-4 rounded-2xl border border-slate-200 bg-slate-50">
              <span className="text-xs font-bold text-navy-900 block mb-2">Accuracy Progression (%)</span>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingCurves} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                    <YAxis domain={[65, 90]} tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="trainAcc" name="Train Accuracy" stroke="#3B82F6" strokeWidth={2} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="valAcc" name="Validation Accuracy" stroke="#008080" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="p-4 rounded-2xl border border-slate-200 bg-slate-50">
              <span className="text-xs font-bold text-navy-900 block mb-2">Loss Progression</span>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingCurves} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0.2, 0.8]} tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#EF4444" strokeWidth={2} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#F59E0B" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 2: UNCERTAINTY & SELECTIVE CLASSIFICATION BENCHMARK (NEW FEATURE) */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-4 border-b border-slate-100">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-700 bg-amber-50 px-2 py-0.5 rounded">
              Clinical Safety Mechanism
            </span>
            <h3 className="text-lg font-bold text-navy-900 mt-1">
              Uncertainty & Selective Classification (Rejection Threshold Analysis)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluated across all 5,687 holdout radiographs to prevent false automated classifications on ambiguous or out-of-distribution inputs.
            </p>
          </div>
          <div className="p-2.5 rounded-xl bg-teal-50 border border-teal-200 text-right">
            <span className="text-[10px] font-bold uppercase text-teal-800 block">Recommended Operational Gate</span>
            <span className="text-base font-extrabold text-teal-900">&tau; = 70.0% (86.11% Acc)</span>
          </div>
        </div>

        {/* Coverage vs Accuracy Chart */}
        <div className="p-4 rounded-2xl border border-slate-200 bg-slate-50">
          <span className="text-xs font-bold text-navy-900 block mb-2">
            Coverage (%) vs. Selective Classification Accuracy (%) Curve
          </span>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={threshCurve} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="threshold" tick={{ fontSize: 10 }} label={{ value: 'Confidence Threshold (tau)', position: 'insideBottom', offset: -5, fontSize: 11 }} />
                <YAxis domain={[60, 100]} tick={{ fontSize: 10 }} unit="%" />
                <Tooltip contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="accuracy" name="Selective Accuracy (%)" stroke="#008080" strokeWidth={2.5} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="macroF1" name="Selective Macro F1 (%)" stroke="#3B82F6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="coverage" name="Coverage (%)" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Threshold Metrics Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs text-left border border-slate-200 rounded-xl overflow-hidden">
            <thead className="bg-navy-900 text-white font-semibold">
              <tr>
                <th className="py-2.5 px-4">Threshold (&tau;)</th>
                <th className="py-2.5 px-4">Coverage</th>
                <th className="py-2.5 px-4">Retained Radiographs</th>
                <th className="py-2.5 px-4">Selective Accuracy</th>
                <th className="py-2.5 px-4">Macro F1</th>
                <th className="py-2.5 px-4">Rejection Rate</th>
                <th className="py-2.5 px-4">Clinical Recommendation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-800">
              {(thresholdData?.thresholds || []).map((row, idx) => (
                <tr key={idx} className={row.threshold === 0.70 ? 'bg-teal-50/80 font-bold text-teal-950' : 'hover:bg-slate-50'}>
                  <td className="py-2 px-4">{Math.round(row.threshold * 100)}%</td>
                  <td className="py-2 px-4">{row.coverage_percent}%</td>
                  <td className="py-2 px-4">{row.retained_count} / 5,687</td>
                  <td className="py-2 px-4 text-teal-800 font-bold">{row.selective_accuracy}%</td>
                  <td className="py-2 px-4">{row.macro_f1}%</td>
                  <td className="py-2 px-4 text-amber-700">{row.rejection_percent}%</td>
                  <td className="py-2 px-4 text-[11px]">
                    {row.threshold === 0.0 && 'Baseline (Forced Closed-World Classification)'}
                    {row.threshold === 0.70 && 'Optimal Balance: +5.5% accuracy gain, 83.1% coverage'}
                    {row.threshold === 0.95 && 'High-Certainty Sub-Cohort (92.5% accuracy)'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* SECTION 3: COVID-19 SUPERVISED SEVERITY BENCHMARKS */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-4 border-b border-slate-100">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded">
              Supervised Severity
            </span>
            <h3 className="text-lg font-bold text-navy-900 mt-1">
              COVID-19 Severity Model Evaluation (Held-Out Test Set)
            </h3>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block uppercase">Test Accuracy</span>
              <span className="text-xl font-extrabold text-blue-700">{severityData?.test_accuracy}%</span>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block uppercase">Macro F1</span>
              <span className="text-xl font-extrabold text-navy-900">{severityData?.macro_f1}%</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Per-class metrics */}
          <div>
            <h4 className="text-xs font-bold text-navy-900 mb-2">Test Split Classification Metrics</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                    <th className="py-2 px-3">Severity Tier</th>
                    <th className="py-2 px-3">Precision</th>
                    <th className="py-2 px-3">Recall</th>
                    <th className="py-2 px-3">F1-Score</th>
                    <th className="py-2 px-3">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {Object.entries(severityData?.per_class || {}).map(([sName, m]) => (
                    <tr key={sName} className="hover:bg-slate-50">
                      <td className="py-2 px-3 font-bold text-navy-900">{sName}</td>
                      <td className="py-2 px-3">{m.precision}%</td>
                      <td className="py-2 px-3">{m.recall}%</td>
                      <td className="py-2 px-3 font-semibold text-blue-800">{m.f1}%</td>
                      <td className="py-2 px-3 text-slate-500">{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Severity Confusion Matrix */}
          <div>
            <h4 className="text-xs font-bold text-navy-900 mb-2">Severity Confusion Matrix (Test Split)</h4>
            <table className="min-w-full text-xs text-center border border-slate-200 rounded-xl overflow-hidden">
              <thead className="bg-navy-800 text-white font-semibold">
                <tr>
                  <th className="py-2 px-3 text-left">Actual \ Predicted</th>
                  {severityData?.confusion_matrix?.labels.map(l => (
                    <th key={l} className="py-2 px-3">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-slate-800">
                {severityData?.confusion_matrix?.matrix.map((row, rIdx) => {
                  const label = severityData.confusion_matrix.labels[rIdx];
                  return (
                    <tr key={label}>
                      <td className="py-2 px-3 text-left font-bold bg-slate-50 border-r border-slate-200">
                        {label}
                      </td>
                      {row.map((val, cIdx) => (
                        <td 
                          key={cIdx} 
                          className={`py-2 px-3 font-semibold ${
                            rIdx === cIdx ? 'bg-blue-50 text-blue-900 font-bold' : val > 0 ? 'text-amber-800' : 'text-slate-300'
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
