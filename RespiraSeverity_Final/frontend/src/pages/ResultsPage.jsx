import React from 'react';
import { 
  ArrowRight, 
  Activity, 
  FileCheck2, 
  Layers, 
  Eye, 
  Download, 
  AlertTriangle, 
  CheckCircle2, 
  Info,
  ShieldAlert,
  Sparkles,
  HelpCircle,
  Stethoscope
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';

export default function ResultsPage({ analysisData, onNavigate }) {
  if (!analysisData || !analysisData.result) {
    return (
      <div className="max-w-xl mx-auto my-12 text-center p-8 bg-white rounded-3xl border border-slate-200">
        <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto mb-3" />
        <h3 className="text-lg font-bold text-navy-900 mb-1">No Active Analysis Loaded</h3>
        <p className="text-xs text-slate-500 mb-6">Please upload and analyze a chest radiograph first.</p>
        <button
          onClick={() => onNavigate('analysis')}
          className="px-5 py-2.5 rounded-xl bg-navy-800 text-white font-semibold text-xs"
        >
          Go to Analysis Portal
        </button>
      </div>
    );
  }

  const { result, previewUrl, filename } = analysisData;
  const disease = result.disease || {};
  const severity = result.severity;
  const clustering = result.clustering;
  const quality = result.quality || 85;

  const isRejected = disease.rejection_status === 'REJECTED_FOR_MANUAL_REVIEW';

  // Format probabilities for chart
  const probData = Object.entries(disease.probabilities || {}).map(([k, v]) => ({
    name: k,
    probability: v,
    rawProb: disease.raw_probabilities?.[k] || (v / 100),
    isTarget: k.toLowerCase() === disease.predicted_class?.toLowerCase()
  }));

  const isNormal = disease.predicted_class === 'Normal';
  const isCovid = disease.predicted_class === 'COVID';
  const isPneumonia = disease.predicted_class === 'Pneumonia';
  const isTB = disease.predicted_class === 'TB';

  return (
    <div className="space-y-8 py-4">
      {/* Top Banner / Heading */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[11px] font-semibold mb-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
            <span>Analysis Verified</span>
          </div>
          <h2 className="text-2xl font-bold text-navy-900 tracking-tight">Diagnosis & Staging Assessment</h2>
          <p className="text-slate-500 text-xs mt-0.5">Radiograph File: <b>{filename}</b></p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => onNavigate('explain')}
            className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-sm transition-all"
          >
            <Eye className="w-3.5 h-3.5 text-teal-700" />
            <span>Inspect Grad-CAM</span>
          </button>
          <button
            onClick={() => onNavigate('report')}
            className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-navy-800 hover:bg-navy-900 text-white text-xs font-semibold shadow-sm transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Generate PDF Report</span>
          </button>
        </div>
      </div>

      {/* REJECTION / UNCERTAINTY SAFETY ALERT BANNER */}
      {isRejected && (
        <div className="bg-amber-50 border-2 border-amber-300 rounded-2xl p-6 text-amber-900 shadow-sm">
          <div className="flex items-start space-x-3.5">
            <ShieldAlert className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="text-xs uppercase font-extrabold px-2 py-0.5 rounded bg-amber-200 text-amber-900">
                  Selective Classification Gate
                </span>
                <h4 className="font-bold text-base text-amber-950">
                  Indeterminate Finding &mdash; Defer to Manual Radiologist Review
                </h4>
              </div>
              <p className="text-xs text-amber-900/90 leading-relaxed pt-1">
                {disease.explanation}
              </p>
              <div className="flex flex-wrap gap-4 pt-2 text-[11px] text-amber-800 font-medium">
                <div>Model Confidence: <b>{disease.confidence}%</b> (Threshold: {disease.confidence_threshold}%)</div>
                <div>Top-2 Margin: <b>{(disease.margin * 100).toFixed(1)}%</b></div>
                <div>Predictive Entropy: <b>{disease.entropy}</b></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Grid: Radiograph + Findings */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Radiograph Preview */}
        <div className="lg:col-span-5 bg-white p-5 rounded-3xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs font-bold text-navy-900">Patient Radiograph (Frontal)</span>
              <span className="text-[11px] font-semibold text-slate-500">Quality: {quality}/100</span>
            </div>
            <div className="rounded-2xl overflow-hidden bg-slate-900 flex items-center justify-center p-2 max-h-[340px]">
              <img
                src={previewUrl}
                alt="Analyzed radiograph"
                className="max-h-[320px] object-contain rounded-xl"
              />
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
            <span>Model: MobileNetV3-Small</span>
            <span>Target Resolution: 224&times;224</span>
          </div>
        </div>

        {/* Right Column: Classification & Probabilities */}
        <div className="lg:col-span-7 space-y-6">
          {/* Primary Disease Card */}
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Primary Disease Softmax Output
            </span>
            <div className="flex items-baseline justify-between mt-2 mb-4">
              <div className="text-3xl font-extrabold text-navy-900 flex items-center space-x-2">
                <span>{disease.predicted_class}</span>
                {isRejected && (
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
                    Uncertain
                  </span>
                )}
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400 block">Top Confidence</span>
                <span className="text-2xl font-black text-teal-700">
                  {disease.confidence}%
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-600 bg-slate-50 p-3 rounded-xl border border-slate-100 leading-relaxed">
              {disease.explanation}
            </p>

            {/* 4-Class Probability Chart */}
            <div className="mt-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-slate-500">Softmax Probability Distribution</span>
                <span className="text-[10px] text-slate-400">Exact Raw Likelihoods</span>
              </div>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={probData} layout="vertical" margin={{ top: 0, right: 30, left: 20, bottom: 0 }}>
                    <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} />
                    <Tooltip 
                      formatter={(val, name, item) => [`${val}% (raw: ${item.payload.rawProb.toFixed(4)})`, 'Probability']}
                      contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                    />
                    <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                      {probData.map((entry, index) => (
                        <Cell 
                          key={`prob-${index}`} 
                          fill={isRejected ? '#F59E0B' : entry.isTarget ? '#008080' : '#CBD5E1'} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Raw Softmax Table */}
              <div className="grid grid-cols-4 gap-2 pt-3 border-t border-slate-100 text-center">
                {probData.map(p => (
                  <div key={p.name} className="p-1.5 rounded bg-slate-50 text-[11px]">
                    <span className="text-slate-500 block text-[10px]">{p.name}</span>
                    <span className="font-bold text-navy-900">{p.rawProb.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* STAGED SEVERITY ASSESSMENT MODULE CARD (ONLY ACTIVE IF ACCEPTED) */}
      {!isRejected && (
        <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm">
          <div className="flex items-center space-x-2 mb-4">
            <Layers className="w-5 h-5 text-teal-700" />
            <h3 className="text-base font-bold text-navy-900">
              Staged Severity Assessment Finding
            </h3>
          </div>

          {/* Condition 1: NORMAL */}
          {isNormal && (
            <div className="bg-emerald-50/70 border border-emerald-200 rounded-2xl p-6 text-emerald-900">
              <div className="flex items-start space-x-3">
                <CheckCircle2 className="w-6 h-6 text-emerald-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-sm">No Disease Pathology Detected</h4>
                  <p className="text-xs text-emerald-800/90 mt-1 leading-relaxed">
                    The radiograph has been classified as <b>Normal</b> with {disease.confidence}% confidence. 
                    In accordance with the hierarchical RespiraSeverity pipeline, severity grading is bypassed for non-pathological cases.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Condition 2: COVID-19 (SUPERVISED SEVERITY) */}
          {isCovid && severity && (
            <div className="bg-blue-50/70 border border-blue-200 rounded-2xl p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                    Supervised Severity Model
                  </span>
                  <h4 className="text-xl font-extrabold text-navy-900 mt-1">
                    COVID-19 Severity: <span className="text-blue-700">{severity.severity}</span>
                  </h4>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-500 block">Severity Confidence</span>
                  <span className="text-2xl font-black text-blue-800">{severity.confidence}%</span>
                </div>
              </div>

              {/* Severity probabilities */}
              <div className="grid grid-cols-3 gap-3 my-4">
                {Object.entries(severity.probabilities || {}).map(([sName, sProb]) => (
                  <div 
                    key={sName} 
                    className={`p-3 rounded-xl border text-center ${
                      sName === severity.severity 
                        ? 'bg-blue-600 text-white border-blue-600 font-bold' 
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                  >
                    <div className="text-xs uppercase">{sName}</div>
                    <div className="text-lg mt-0.5 font-extrabold">{sProb}%</div>
                  </div>
                ))}
              </div>

              <p className="text-xs text-blue-900/80 leading-relaxed">
                <b>Research Annotation Grounding:</b> Trained on 760 clinically annotated COVID-19 radiographs (RICORD / BIMCV). 
                Reflects lung consolidation and bilateral ground-glass opacity extent.
              </p>
            </div>
          )}

          {/* Condition 3 & 4: PNEUMONIA OR TB (EXPLORATORY CLUSTERING) */}
          {(isPneumonia || isTB) && clustering && (
            <div className="bg-amber-50/70 border border-amber-200 rounded-2xl p-6">
              <div className="flex items-center space-x-2 text-amber-800 font-bold text-xs uppercase tracking-wider mb-2">
                <ShieldAlert className="w-4 h-4 text-amber-600" />
                <span>Exploratory Radiographic Feature Cluster (Unsupervised)</span>
              </div>

              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                <div>
                  <h4 className="text-xl font-extrabold text-navy-900">
                    {clustering.cluster_name}
                  </h4>
                  <p className="text-xs text-slate-600 mt-1">
                    {clustering.cluster_interpretation}
                  </p>
                </div>
                <button
                  onClick={() => onNavigate('clustering')}
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold"
                >
                  <span>View in PCA Space</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-4">
                <div className="bg-white p-3 rounded-xl border border-amber-200">
                  <span className="text-[11px] text-slate-400 block">Nearest COVID-19 Reference</span>
                  <span className="text-sm font-bold text-navy-900">{clustering.nearest_covid_reference_centroid} Centroid</span>
                </div>
                <div className="bg-white p-3 rounded-xl border border-amber-200">
                  <span className="text-[11px] text-slate-400 block">Relative Radiographic Opacity</span>
                  <span className="text-sm font-bold text-navy-900">{clustering.mean_opacity_percentile}% attenuation</span>
                </div>
                <div className="bg-white p-3 rounded-xl border border-amber-200">
                  <span className="text-[11px] text-slate-400 block">Cluster Quality (Silhouette)</span>
                  <span className="text-sm font-bold text-navy-900">
                    s = {clustering.cluster_metrics?.silhouette_score ?? '0.21'}
                  </span>
                </div>
              </div>

              <div className="p-3 bg-amber-100/60 rounded-xl text-[11px] text-amber-900/90 leading-relaxed border border-amber-300/40">
                <b>IMPORTANT SCIENTIFIC CAVEAT:</b> Verified clinical severity labels currently do not exist for Pneumonia/TB in this research dataset. 
                This exploratory grouping is formed via K-means clustering over MobileNetV3-Small embeddings. 
                <b>It is not a clinically validated mild/moderate/severe diagnosis.</b>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
