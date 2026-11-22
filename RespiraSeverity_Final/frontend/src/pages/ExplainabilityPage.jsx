import React, { useState } from 'react';
import { 
  Eye, 
  Layers, 
  Info, 
  ShieldAlert, 
  CheckCircle2, 
  Maximize2,
  SlidersHorizontal
} from 'lucide-react';

export default function ExplainabilityPage({ analysisData, onNavigate }) {
  const [activeView, setActiveView] = useState('overlay'); // 'original', 'heatmap', 'overlay', 'side-by-side'

  if (!analysisData || !analysisData.result) {
    return (
      <div className="max-w-xl mx-auto my-12 text-center p-8 bg-white rounded-3xl border border-slate-200">
        <Info className="w-12 h-12 text-teal-700 mx-auto mb-3" />
        <h3 className="text-lg font-bold text-navy-900 mb-1">No Active Analysis to Explain</h3>
        <p className="text-xs text-slate-500 mb-6">Upload and process a chest radiograph to view its Grad-CAM attention heatmap.</p>
        <button
          onClick={() => onNavigate('analysis')}
          className="px-5 py-2.5 rounded-xl bg-navy-800 text-white font-semibold text-xs"
        >
          Go to Analysis Portal
        </button>
      </div>
    );
  }

  const gradcam = analysisData.result.gradcam;
  const disease = analysisData.result.disease || {};

  if (!gradcam) {
    return (
      <div className="max-w-xl mx-auto my-12 text-center p-8 bg-white rounded-3xl border border-slate-200">
        <ShieldAlert className="w-12 h-12 text-amber-500 mx-auto mb-3" />
        <h3 className="text-lg font-bold text-navy-900 mb-1">Grad-CAM Map Unavailable</h3>
        <p className="text-xs text-slate-500">Grad-CAM visualization was not generated for this image.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[11px] font-semibold mb-1">
            <Eye className="w-3.5 h-3.5 text-teal-600" />
            <span>Interpretability Module</span>
          </div>
          <h2 className="text-2xl font-bold text-navy-900 tracking-tight">
            Grad-CAM Visual Attribution
          </h2>
          <p className="text-slate-500 text-xs mt-0.5">
            Target Predicted Class: <b className="text-navy-900">{disease.predicted_class}</b> ({disease.confidence}% confidence)
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center space-x-1 p-1 bg-slate-100 rounded-xl border border-slate-200 text-xs">
          <button
            onClick={() => setActiveView('overlay')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeView === 'overlay' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Diagnostic Overlay
          </button>
          <button
            onClick={() => setActiveView('heatmap')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeView === 'heatmap' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Heatmap Only
          </button>
          <button
            onClick={() => setActiveView('original')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeView === 'original' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Original Radiograph
          </button>
          <button
            onClick={() => setActiveView('side-by-side')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeView === 'side-by-side' ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
            }`}
          >
            Side-by-Side
          </button>
        </div>
      </div>

      {/* Main Image Viewer */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm">
        {activeView !== 'side-by-side' ? (
          <div className="flex flex-col items-center">
            <div className="w-full max-w-lg rounded-2xl overflow-hidden bg-slate-950 p-2 flex items-center justify-center">
              <img
                src={
                  activeView === 'overlay' ? gradcam.overlay_base64 :
                  activeView === 'heatmap' ? gradcam.heatmap_base64 :
                  gradcam.original_base64
                }
                alt={`Grad-CAM ${activeView}`}
                className="w-full h-auto max-h-[460px] object-contain rounded-xl"
              />
            </div>
            <span className="text-xs font-semibold text-slate-500 mt-3">
              {activeView === 'overlay' && 'High-activation gradients overlaid onto original anatomical thoracic structures (Jet Colormap, α = 0.45)'}
              {activeView === 'heatmap' && 'Normalized gradient class activation map isolated from background anatomy'}
              {activeView === 'original' && 'Original 224×224 normalized frontal radiograph input'}
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2 text-center">
              <div className="text-xs font-bold text-navy-900">Original Radiograph</div>
              <div className="rounded-xl overflow-hidden bg-slate-950 p-1.5">
                <img src={gradcam.original_base64} alt="Original" className="w-full h-auto object-contain rounded-lg" />
              </div>
            </div>
            <div className="space-y-2 text-center">
              <div className="text-xs font-bold text-navy-900">Activation Heatmap</div>
              <div className="rounded-xl overflow-hidden bg-slate-950 p-1.5">
                <img src={gradcam.heatmap_base64} alt="Heatmap" className="w-full h-auto object-contain rounded-lg" />
              </div>
            </div>
            <div className="space-y-2 text-center">
              <div className="text-xs font-bold text-navy-900">Anatomical Overlay</div>
              <div className="rounded-xl overflow-hidden bg-slate-950 p-1.5">
                <img src={gradcam.overlay_base64} alt="Overlay" className="w-full h-auto object-contain rounded-lg" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Explanatory Technical Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="flex items-center space-x-2 text-navy-900 font-bold text-sm mb-2">
            <Layers className="w-4 h-4 text-teal-700" />
            <span>Target Convolutional Layer</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mb-3">
            Gradients are backpropagated into the final inverted bottleneck residual block of MobileNetV3-Small:
          </p>
          <div className="p-2.5 rounded-lg bg-slate-100 font-mono text-[11px] text-navy-900 border border-slate-200">
            {gradcam.target_layer}
          </div>
          <p className="text-[11px] text-slate-500 mt-3 leading-relaxed">
            This penultimate convolutional representation captures abstract semantic features (e.g. consolidations, pleural thickening, apical infiltrates) prior to global average pooling.
          </p>
        </div>

        <div className="bg-amber-50/70 p-6 rounded-2xl border border-amber-200 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-amber-900 font-bold text-sm mb-2">
              <ShieldAlert className="w-4 h-4 text-amber-700" />
              <span>Scientific Interpretability Disclaimer</span>
            </div>
            <p className="text-xs text-amber-900/90 leading-relaxed">
              Grad-CAM reveals the receptive field regions that most strongly conditioned the model's logits. 
              <b>It does not constitute localized radiologist ground truth or definitive anatomical segmentation.</b>
            </p>
          </div>
          <div className="text-[10px] text-amber-800/80 mt-4 pt-3 border-t border-amber-200/60">
            Reference: Selvaraju et al., &ldquo;Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,&rdquo; ICCV 2017.
          </div>
        </div>
      </div>
    </div>
  );
}
