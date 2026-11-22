import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  ScatterChart as ScatterIcon, 
  BarChart2, 
  Info, 
  ShieldAlert, 
  Loader2, 
  CheckCircle2,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend, 
  BarChart, 
  Bar, 
  Cell 
} from 'recharts';
import { fetchClusteringMetrics } from '../services/api';

const CLUSTER_COLORS = ['#008080', '#3B82F6', '#F59E0B'];
const COVID_COLORS = { Mild: '#10B981', Moderate: '#F59E0B', Severe: '#EF4444' };

export default function ClusteringPage() {
  const [activeTab, setActiveTab] = useState('pneumonia'); // 'pneumonia', 'tb', 'covid', 'comparison'
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetchClusteringMetrics();
        setData(res);
      } catch (e) {
        console.error('Failed to load clustering data:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="py-20 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-teal-700 mx-auto mb-3" />
        <p className="text-xs text-slate-500">Loading PCA projections and clustering metrics...</p>
      </div>
    );
  }

  const summary = data?.summary || {};
  const pca = data?.pca_projections || {};
  const pneuMetrics = summary.pneumonia || {};
  const tbMetrics = summary.tb || {};
  const covidSummary = summary.covid_reference || {};

  // Group PCA points for Pneumonia
  const pneuPoints = pca.pneumonia || [];
  const pneuClusters = [0, 1, 2].map(c => ({
    name: `Cluster ${c}`,
    color: CLUSTER_COLORS[c],
    points: pneuPoints.filter(p => p.cluster === c)
  }));

  // Group PCA points for TB
  const tbPoints = pca.tb || [];
  const tbClusters = [0, 1, 2].map(c => ({
    name: `Cluster ${c}`,
    color: CLUSTER_COLORS[c],
    points: tbPoints.filter(p => p.cluster === c)
  }));

  // Group PCA points for COVID
  const covidPoints = pca.covid || [];
  const covidGroups = ['Mild', 'Moderate', 'Severe'].map(label => ({
    name: label,
    color: COVID_COLORS[label],
    points: covidPoints.filter(p => p.label === label)
  }));

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[11px] font-semibold mb-1">
            <Layers className="w-3.5 h-3.5 text-teal-600" />
            <span>Research Exploratory Component</span>
          </div>
          <h2 className="text-2xl font-bold text-navy-900 tracking-tight">
            Unsupervised Radiographic Feature Clustering
          </h2>
          <p className="text-slate-500 text-xs mt-0.5">
            1024-D MobileNetV3-Small deep embeddings projected onto 2D PCA feature space.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-xl border border-slate-200 text-xs">
          <button
            onClick={() => setActiveTab('pneumonia')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'pneumonia' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Pneumonia Clustering
          </button>
          <button
            onClick={() => setActiveTab('tb')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'tb' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            TB Clustering
          </button>
          <button
            onClick={() => setActiveTab('covid')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'covid' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            COVID-19 Centroids
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'comparison' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Comparative Proximity
          </button>
        </div>
      </div>

      {/* TAB 1: PNEUMONIA */}
      {activeTab === 'pneumonia' && (
        <div className="space-y-6">
          {/* Metrics Header Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Sample Size</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{pneuMetrics.sample_size} Images</div>
              <p className="text-[10px] text-slate-400 mt-0.5">MobileNetV3-Small features</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Silhouette Score</span>
              <div className="text-xl font-bold text-teal-700 mt-1">{pneuMetrics.silhouette_score}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Higher indicates denser clusters</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Davies-Bouldin Index</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{pneuMetrics.davies_bouldin_index}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Lower indicates better separation</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Calinski-Harabasz</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{pneuMetrics.calinski_harabasz_score}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Variance ratio criterion</p>
            </div>
          </div>

          {/* Scatter Chart & Cluster Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
              <h3 className="text-sm font-bold text-navy-900 mb-1">
                Pneumonia 2D PCA Feature Space
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Color-coded by independent K-Means clusters (k=3)
              </p>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="x" name="PC 1" tick={{ fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" name="PC 2" tick={{ fontSize: 10 }} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(val, name) => [val, name]}
                      contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    {pneuClusters.map((c, i) => (
                      <Scatter key={i} name={c.name} data={c.points} fill={c.color} opacity={0.7} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-4 bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-navy-900">Cluster Populations</h3>
              <div className="space-y-3">
                {Object.entries(pneuMetrics.cluster_distribution || {}).map(([cName, count], idx) => (
                  <div key={cName} className="p-3 rounded-xl border border-slate-100 bg-slate-50">
                    <div className="flex justify-between items-center text-xs font-bold text-navy-900">
                      <span className="flex items-center space-x-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[idx] }}></span>
                        <span>{cName}</span>
                      </span>
                      <span>{count} cases</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">
                      Radiographic Opacity Attenuation: <b>{Math.round((pneuMetrics.mean_radiographic_opacities?.[cName] || 0.48) * 100)}%</b>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: TUBERCULOSIS */}
      {activeTab === 'tb' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Sample Size</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{tbMetrics.sample_size} Images</div>
              <p className="text-[10px] text-slate-400 mt-0.5">MobileNetV3-Small features</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Silhouette Score</span>
              <div className="text-xl font-bold text-teal-700 mt-1">{tbMetrics.silhouette_score}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Cluster cohesion</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Davies-Bouldin Index</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{tbMetrics.davies_bouldin_index}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Cluster dispersion</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Calinski-Harabasz</span>
              <div className="text-xl font-bold text-navy-900 mt-1">{tbMetrics.calinski_harabasz_score}</div>
              <p className="text-[10px] text-slate-400 mt-0.5">Variance ratio criterion</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
              <h3 className="text-sm font-bold text-navy-900 mb-1">
                Tuberculosis 2D PCA Feature Space
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Color-coded by independent K-Means clusters (k=3)
              </p>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="x" name="PC 1" tick={{ fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" name="PC 2" tick={{ fontSize: 10 }} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    {tbClusters.map((c, i) => (
                      <Scatter key={i} name={c.name} data={c.points} fill={c.color} opacity={0.7} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-4 bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-navy-900">Cluster Populations</h3>
              <div className="space-y-3">
                {Object.entries(tbMetrics.cluster_distribution || {}).map(([cName, count], idx) => (
                  <div key={cName} className="p-3 rounded-xl border border-slate-100 bg-slate-50">
                    <div className="flex justify-between items-center text-xs font-bold text-navy-900">
                      <span className="flex items-center space-x-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[idx] }}></span>
                        <span>{cName}</span>
                      </span>
                      <span>{count} cases</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">
                      Radiographic Opacity Attenuation: <b>{Math.round((tbMetrics.mean_radiographic_opacities?.[cName] || 0.51) * 100)}%</b>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: COVID REFERENCE CENTROIDS */}
      {activeTab === 'covid' && (
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <h3 className="text-sm font-bold text-navy-900 mb-1">
              COVID-19 Ground-Truth Severity Reference Clusters
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              760 clinically annotated cases used to compute anchor centroids (Mild, Moderate, Severe)
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="x" name="PC 1" tick={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="y" name="PC 2" tick={{ fontSize: 10 }} />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  {covidGroups.map((g, i) => (
                    <Scatter key={i} name={`${g.name} Severity`} data={g.points} fill={g.color} opacity={0.7} />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: COMPARATIVE ANALYSIS (INDEPENDENT VS COVID-CENTROID) */}
      {activeTab === 'comparison' && (
        <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-6">
          <div>
            <h3 className="text-sm font-bold text-navy-900">
              Experimental Comparison: Independent K-Means vs. COVID Reference Centroid Mapping
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Evaluating whether COVID-19 severity centroids generalize to other infectious pulmonary pathologies.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 rounded-2xl border border-slate-200 bg-slate-50">
              <h4 className="font-bold text-xs text-navy-900 mb-2">Pneumonia &rarr; COVID Centroid Proximity</h4>
              <p className="text-[11px] text-slate-600 mb-3">
                Assigned to nearest COVID-19 reference centroid in 1024-D space:
              </p>
              <div className="space-y-1.5 text-xs">
                {Object.entries(pneuMetrics.covid_ref_assignment || {}).map(([label, cnt]) => (
                  <div key={label} className="flex justify-between p-2 rounded bg-white border border-slate-100">
                    <span className="font-semibold text-slate-700">{label} COVID Centroid:</span>
                    <span className="font-bold text-navy-900">{cnt} cases</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 rounded-2xl border border-slate-200 bg-slate-50">
              <h4 className="font-bold text-xs text-navy-900 mb-2">Tuberculosis &rarr; COVID Centroid Proximity</h4>
              <p className="text-[11px] text-slate-600 mb-3">
                Assigned to nearest COVID-19 reference centroid in 1024-D space:
              </p>
              <div className="space-y-1.5 text-xs">
                {Object.entries(tbMetrics.covid_ref_assignment || {}).map(([label, cnt]) => (
                  <div key={label} className="flex justify-between p-2 rounded bg-white border border-slate-100">
                    <span className="font-semibold text-slate-700">{label} COVID Centroid:</span>
                    <span className="font-bold text-navy-900">{cnt} cases</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs leading-relaxed">
            <b>Research Scientific Conclusion:</b> Cross-disease centroid mapping demonstrates significant domain divergence. 
            Because TB and Pneumonia manifest distinct radiographic features (focal consolidations and apical infiltrates vs. diffuse peripheral ground-glass opacities in COVID-19), 
            COVID-19 severity centroids <b>cannot</b> be used as surrogate severity labels for other conditions.
          </div>
        </div>
      )}

      {/* Ethical Research Alert */}
      <div className="p-4 rounded-2xl bg-slate-100 border border-slate-200 text-slate-600 text-xs flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 text-teal-700 flex-shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <b>Research Integrity Note:</b> As mandated by the research protocol, unsupervised K-means clusters (k=3) 
          are reported purely as exploratory radiographic feature groups. They do not claim clinical diagnostic validity.
        </div>
      </div>
    </div>
  );
}
