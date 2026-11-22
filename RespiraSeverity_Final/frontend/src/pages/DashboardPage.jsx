import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Layers, 
  Activity, 
  CheckCircle2, 
  Clock, 
  TrendingUp, 
  ArrowUpRight,
  ShieldCheck,
  Server
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  PieChart, 
  Pie 
} from 'recharts';
import { fetchModelInfo } from '../services/api';

const DATASET_DIST = [
  { name: 'Tuberculosis', count: 10641, percentage: 37.42, color: '#0F2942' },
  { name: 'Normal', count: 9088, percentage: 31.96, color: '#008080' },
  { name: 'Pneumonia', count: 5824, percentage: 20.48, color: '#3B82F6' },
  { name: 'COVID-19', count: 2881, percentage: 10.13, color: '#F59E0B' },
];

const SEVERITY_DIST = [
  { name: 'Moderate', count: 284, percentage: 37.37, color: '#F59E0B' },
  { name: 'Severe', count: 242, percentage: 31.84, color: '#EF4444' },
  { name: 'Mild', count: 234, percentage: 30.79, color: '#10B981' },
];

export default function DashboardPage({ onNavigate, history = [] }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const info = await fetchModelInfo();
        setModelInfo(info);
      } catch (e) {
        console.error('Error fetching model info:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-navy-900 tracking-tight">Research Analytics Dashboard</h2>
          <p className="text-slate-500 text-xs mt-1">
            Real-time pipeline metrics, model statuses, and multi-source dataset distribution.
          </p>
        </div>
        <button
          onClick={() => onNavigate('analysis')}
          className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-navy-800 hover:bg-navy-900 text-white text-xs font-semibold shadow-sm transition-all"
        >
          <span>Upload New Radiograph</span>
          <ArrowUpRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Images */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Total Training Data</span>
            <div className="p-2 rounded-lg bg-navy-50 text-navy-800">
              <Database className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-navy-900">28,434</div>
          <p className="text-[11px] text-slate-500 mt-1">Across 3 benchmark repositories</p>
        </div>

        {/* Primary Accuracy */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Disease Accuracy</span>
            <div className="p-2 rounded-lg bg-teal-50 text-teal-700">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-teal-700">80.62%</div>
          <p className="text-[11px] text-slate-500 mt-1">Macro F1-Score: <b>84.39%</b></p>
        </div>

        {/* Backbone Architecture */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">CNN Backbone</span>
            <div className="p-2 rounded-lg bg-indigo-50 text-indigo-700">
              <Server className="w-4 h-4" />
            </div>
          </div>
          <div className="text-base font-extrabold text-navy-900">MobileNetV3-Small</div>
          <p className="text-[11px] text-slate-500 mt-1">Pretrained ImageNet Transfer Weights</p>
        </div>

        {/* COVID Severity Labelled */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Severity Annotations</span>
            <div className="p-2 rounded-lg bg-amber-50 text-amber-700">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-amber-700">760 Cases</div>
          <p className="text-[11px] text-slate-500 mt-1">Mild: 234 &bull; Mod: 284 &bull; Sev: 242</p>
        </div>
      </div>

      {/* Model Status Checklist */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <h3 className="text-sm font-bold text-navy-900 mb-4 flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-teal-700" />
          <span>Active Pipeline Model Modules</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-emerald-100 bg-emerald-50/50 flex items-start space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-navy-900">Disease Classifier (4-Class)</div>
              <p className="text-[11px] text-slate-600 mt-0.5">MobileNetV3-Small &bull; 80.62% Baseline</p>
              <span className="inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                Active in Memory
              </span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-emerald-100 bg-emerald-50/50 flex items-start space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-navy-900">COVID-19 Severity Model</div>
              <p className="text-[11px] text-slate-600 mt-0.5">Supervised 3-Tier Staging &bull; 760 Cases</p>
              <span className="inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                Active in Memory
              </span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-teal-100 bg-teal-50/50 flex items-start space-x-3">
            <CheckCircle2 className="w-5 h-5 text-teal-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-navy-900">Exploratory Clustering Module</div>
              <p className="text-[11px] text-slate-600 mt-0.5">Pneumonia & TB K-Means (k=3) + PCA</p>
              <span className="inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded bg-teal-100 text-teal-800">
                Active in Memory
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Breakdown Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Disease Distribution Bar Chart */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-navy-900 mb-1">Disease Dataset Distribution</h3>
          <p className="text-xs text-slate-500 mb-6">Total of 28,434 images across four target conditions</p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={DATASET_DIST} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip 
                  formatter={(value, name, item) => [`${value} images (${item.payload.percentage}%)`, 'Count']}
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {DATASET_DIST.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Severity Distribution Pie Chart */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-navy-900 mb-1">COVID-19 Severity Dataset Distribution</h3>
          <p className="text-xs text-slate-500 mb-6">760 clinically annotated cases across Mild, Moderate, and Severe</p>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={SEVERITY_DIST}
                  dataKey="count"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  innerRadius={45}
                  paddingAngle={4}
                  label={({ name, percentage }) => `${name} (${percentage}%)`}
                >
                  {SEVERITY_DIST.map((entry, index) => (
                    <Cell key={`sev-cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value, name, item) => [`${value} cases (${item.payload.percentage}%)`, name]}
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Predictions History */}
      {history.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-navy-900 flex items-center space-x-2">
              <Clock className="w-4 h-4 text-slate-500" />
              <span>Session Analysis History</span>
            </h3>
            <span className="text-xs text-slate-400">{history.length} analyzed in this session</span>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold">
                  <th className="py-2 px-3">Timestamp</th>
                  <th className="py-2 px-3">File / ID</th>
                  <th className="py-2 px-3">Predicted Disease</th>
                  <th className="py-2 px-3">Confidence</th>
                  <th className="py-2 px-3">Severity / Staging</th>
                  <th className="py-2 px-3">Quality</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {history.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    <td className="py-2 px-3 text-slate-500">{item.timestamp}</td>
                    <td className="py-2 px-3 font-medium text-navy-900">{item.filename}</td>
                    <td className="py-2 px-3">
                      <span className="px-2 py-0.5 rounded-full font-semibold bg-navy-50 text-navy-800">
                        {item.disease}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-medium">{item.confidence}%</td>
                    <td className="py-2 px-3">
                      {item.severity ? (
                        <span className="px-2 py-0.5 rounded-full bg-blue-50 text-blue-800 font-medium">
                          {item.severity}
                        </span>
                      ) : item.cluster ? (
                        <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 font-medium">
                          {item.cluster} (Exploratory)
                        </span>
                      ) : (
                        <span className="text-slate-400">N/A</span>
                      )}
                    </td>
                    <td className="py-2 px-3">{item.quality}/100</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
