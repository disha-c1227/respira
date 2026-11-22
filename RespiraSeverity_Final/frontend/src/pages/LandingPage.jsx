import React from 'react';
import { 
  ArrowRight, 
  Activity, 
  Layers, 
  ShieldCheck, 
  Eye, 
  FileCheck2, 
  Cpu, 
  Database,
  GitBranch,
  Stethoscope
} from 'lucide-react';

export default function LandingPage({ onStartAnalysis, onNavigate }) {
  return (
    <div className="space-y-16 py-6">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-navy-900 via-navy-800 to-teal-900 text-white p-8 sm:p-12 lg:p-16 shadow-xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-teal-500/20 text-teal-300 border border-teal-400/30 text-xs font-semibold uppercase tracking-wider mb-6">
            <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span>
            <span>Final Year Academic Research Project</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight text-white mb-6">
            Respira<span className="text-teal-400">Severity</span>
          </h1>
          <p className="text-base sm:text-xl text-slate-200 font-normal leading-relaxed mb-8">
            A Deep Learning-Based Framework for <b>Pulmonary Disease Classification</b> and <b>Staged Severity Assessment</b> Using Chest Radiographs.
          </p>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={onStartAnalysis}
              className="flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-900 font-bold text-sm transition-all shadow-lg shadow-teal-500/25 hover:shadow-teal-500/40 hover:-translate-y-0.5"
            >
              <span>Start X-Ray Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('dashboard')}
              className="px-6 py-3.5 rounded-xl bg-white/10 hover:bg-white/20 text-white border border-white/20 font-medium text-sm transition-all"
            >
              View Research Dashboard
            </button>
          </div>
        </div>

        {/* Subtle decorative background accents */}
        <div className="absolute right-0 top-0 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>
        <div className="absolute right-12 bottom-8 opacity-10 hidden lg:block">
          <Activity className="w-80 h-80 text-white" />
        </div>
      </section>

      {/* The Scientific Core / Problem & Solution */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-navy-50 text-navy-800 flex items-center justify-center mb-4">
            <GitBranch className="w-6 h-6 text-navy-700" />
          </div>
          <h3 className="text-lg font-bold text-navy-900 mb-2">Hierarchical Staged Routing</h3>
          <p className="text-slate-600 text-sm leading-relaxed">
            Existing diagnostic classifiers stop at binary detection. RespiraSeverity separates primary disease identification from severity grading so condition-specific criteria are respected.
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-teal-50 text-teal-700 flex items-center justify-center mb-4">
            <Stethoscope className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-navy-900 mb-2">Supervised vs. Exploratory</h3>
          <p className="text-slate-600 text-sm leading-relaxed">
            Maintains strict scientific ethics: COVID-19 severity is predicted via supervised labels (760 cases), while Pneumonia and TB utilize exploratory feature clustering (k=3).
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-700 flex items-center justify-center mb-4">
            <Eye className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-navy-900 mb-2">Grad-CAM Explainability</h3>
          <p className="text-slate-600 text-sm leading-relaxed">
            Transparent visual attribution maps overlay gradients directly on the input radiograph, allowing auditing of high-activation lung fields before final reports are created.
          </p>
        </div>
      </section>

      {/* System Pipeline Architecture */}
      <section className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm">
        <div className="text-center max-w-2xl mx-auto mb-10">
          <h2 className="text-2xl font-bold text-navy-900">End-to-End System Architecture</h2>
          <p className="text-slate-500 text-sm mt-2">
            Multi-stage pipeline following the methodology published in our research paper.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
            <span className="w-7 h-7 rounded-full bg-navy-800 text-white inline-flex items-center justify-center text-xs font-bold mb-3">1</span>
            <h4 className="font-bold text-navy-900 text-sm mb-1">Quality & Preprocessing</h4>
            <p className="text-xs text-slate-500">Laplacian blur, contrast variance, luminance verification & 224&times;224 normalization.</p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
            <span className="w-7 h-7 rounded-full bg-navy-800 text-white inline-flex items-center justify-center text-xs font-bold mb-3">2</span>
            <h4 className="font-bold text-navy-900 text-sm mb-1">Disease Classification</h4>
            <p className="text-xs text-slate-500">MobileNetV3-Small backbone predicting Normal, Pneumonia, Tuberculosis, or COVID-19.</p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
            <span className="w-7 h-7 rounded-full bg-navy-800 text-white inline-flex items-center justify-center text-xs font-bold mb-3">3</span>
            <h4 className="font-bold text-navy-900 text-sm mb-1">Severity Assessment</h4>
            <p className="text-xs text-slate-500">Supervised 3-tier COVID staging or exploratory feature clustering for Pneumonia/TB.</p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
            <span className="w-7 h-7 rounded-full bg-teal-700 text-white inline-flex items-center justify-center text-xs font-bold mb-3">4</span>
            <h4 className="font-bold text-navy-900 text-sm mb-1">Explainability & Report</h4>
            <p className="text-xs text-slate-500">Grad-CAM heatmap generation and structured downloadable PDF clinical research summary.</p>
          </div>
        </div>
      </section>

      {/* Dataset & Performance Snapshot */}
      <section className="bg-navy-900 rounded-2xl text-white p-8 shadow-sm">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div>
            <div className="text-3xl sm:text-4xl font-extrabold text-teal-400">28,434</div>
            <div className="text-xs text-slate-300 mt-1 uppercase tracking-wider">Validated Radiographs</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-extrabold text-white">80.62%</div>
            <div className="text-xs text-slate-300 mt-1 uppercase tracking-wider">Disease Accuracy</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-extrabold text-teal-400">84.39%</div>
            <div className="text-xs text-slate-300 mt-1 uppercase tracking-wider">Macro F1-Score</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-extrabold text-white">760</div>
            <div className="text-xs text-slate-300 mt-1 uppercase tracking-wider">Annotated COVID Cases</div>
          </div>
        </div>
      </section>
    </div>
  );
}
