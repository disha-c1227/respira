import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  Image as ImageIcon, 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  Loader2, 
  FileCheck2, 
  Sparkles,
  ArrowRight,
  Sliders,
  ShieldCheck
} from 'lucide-react';
import { runUnifiedPredict, validateImage } from '../services/api';

const SAMPLE_XRAYS = [
  { name: 'Normal Sample', class: 'Normal', path: '/samples/normal_sample.jpg', desc: 'Clear lung fields, sharp costophrenic angles' },
  { name: 'Pneumonia Sample', class: 'Pneumonia', path: '/samples/pneumonia_sample.jpg', desc: 'Focal lobar consolidation' },
  { name: 'Tuberculosis Sample', class: 'TB', path: '/samples/tb_sample.jpg', desc: 'Apical infiltrates and fibro-cavitary lesions' },
  { name: 'COVID-19 Sample', class: 'COVID', path: '/samples/covid_sample.png', desc: 'Peripheral ground-glass opacities' },
];

export default function AnalysisPage({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [threshold, setThreshold] = useState(70.0); // 70% confidence rejection gate
  const fileInputRef = useRef(null);

  // Handle local file selection
  const handleFileChange = async (selectedFile) => {
    if (!selectedFile) return;
    setErrorMsg(null);
    setValidationResult(null);

    // Validate type
    if (!selectedFile.type.match(/^image\/(jpeg|png|jpg)$/i) && !selectedFile.name.match(/\.(jpg|jpeg|png)$/i)) {
      setErrorMsg('Unsupported file format. Please upload a chest X-ray in JPG, JPEG, or PNG format.');
      return;
    }

    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    // Run preliminary client-side image validation check via API
    try {
      const val = await validateImage(selectedFile);
      setValidationResult(val);
      if (!val.valid) {
        setErrorMsg(val.error || 'Image did not pass technical quality criteria.');
      }
    } catch (e) {
      console.warn('Pre-validation check note:', e);
    }
  };

  // Load preset demonstration sample
  const handleSelectSample = async (sample) => {
    try {
      setErrorMsg(null);
      const res = await fetch(sample.path);
      const blob = await res.blob();
      const ext = sample.path.endsWith('.png') ? 'png' : 'jpg';
      const sampleFile = new File([blob], `${sample.class.toLowerCase()}_sample.${ext}`, { type: blob.type || 'image/jpeg' });
      await handleFileChange(sampleFile);
    } catch (e) {
      setErrorMsg(`Could not load sample image: ${e.message}`);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setValidationResult(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setErrorMsg(null);

    try {
      setAnalysisStep('1/4: Validating radiographic quality & focus...');
      await new Promise(r => setTimeout(r, 250));

      setAnalysisStep('2/4: Applying exact grayscale normalization & MobileNetV3...');
      await new Promise(r => setTimeout(r, 250));

      setAnalysisStep(`3/4: Selective classification check (threshold = ${threshold}%)...`);
      const result = await runUnifiedPredict(file, threshold);

      setAnalysisStep('4/4: Generating Grad-CAM explainability...');
      await new Promise(r => setTimeout(r, 250));

      onAnalysisComplete({
        result,
        file,
        previewUrl,
        timestamp: new Date().toLocaleTimeString(),
        filename: file.name
      });
    } catch (err) {
      setErrorMsg(err.message || 'Analysis encountered an error');
      setAnalyzing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-2xl sm:text-3xl font-extrabold text-navy-900 tracking-tight">
          Chest Radiograph Analysis Portal
        </h2>
        <p className="text-slate-500 text-xs sm:text-sm mt-2 max-w-xl mx-auto">
          Upload a frontal view (PA/AP) chest X-ray image for automated four-class pulmonary disease classification, condition-specific severity grading, and Grad-CAM interpretability.
        </p>
      </div>

      {/* Demo Sample Quick Selector */}
      <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm">
        <div className="flex items-center space-x-2 mb-3">
          <Sparkles className="w-4 h-4 text-teal-700" />
          <span className="text-xs font-bold text-navy-900 uppercase tracking-wider">Quick Project Demo Samples</span>
          <span className="text-[10px] text-slate-400">(Click to load verified test cases)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {SAMPLE_XRAYS.map((s, idx) => (
            <button
              key={idx}
              onClick={() => handleSelectSample(s)}
              disabled={analyzing}
              className="text-left p-2.5 rounded-xl border border-slate-200 hover:border-teal-500 hover:bg-teal-50/50 transition-all group disabled:opacity-50"
            >
              <div className="text-xs font-bold text-navy-900 group-hover:text-teal-700">{s.name}</div>
              <div className="text-[10px] text-slate-500 truncate">{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Selective Rejection Threshold Setting Bar */}
      <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-teal-700" />
          <div>
            <span className="text-xs font-bold text-navy-900 block">Uncertainty & Rejection Gate</span>
            <span className="text-[11px] text-slate-500">Defers predictions below confidence threshold to manual radiologist review</span>
          </div>
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto">
          <span className="text-xs text-slate-500 font-medium">Confidence Threshold:</span>
          <select 
            value={threshold} 
            onChange={e => setThreshold(Number(e.target.value))}
            className="text-xs font-bold bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-navy-900 focus:outline-none focus:ring-2 focus:ring-teal-500"
          >
            <option value={0}>0% (Rejection Off)</option>
            <option value={50}>50% Minimal</option>
            <option value={60}>60% Moderate</option>
            <option value={70}>70% Recommended (86.1% Acc)</option>
            <option value={75}>75% Conservative (86.6% Acc)</option>
            <option value={80}>80% Strict (87.2% Acc)</option>
            <option value={90}>90% Ultra-High (89.6% Acc)</option>
          </select>
        </div>
      </div>

      {/* Main Upload Box */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-8">
        {!previewUrl ? (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileChange(e.dataTransfer.files[0]);
              }
            }}
            className="border-2 border-dashed border-slate-300 hover:border-teal-500 rounded-2xl p-8 sm:p-12 text-center transition-colors cursor-pointer bg-slate-50/50 hover:bg-teal-50/20"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
            <div className="w-16 h-16 rounded-2xl bg-teal-50 text-teal-700 flex items-center justify-center mx-auto mb-4 shadow-sm">
              <UploadCloud className="w-8 h-8" />
            </div>
            <h3 className="text-base font-bold text-navy-900 mb-1">
              Drag & drop chest X-ray image here
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Supports standard high-resolution JPG, JPEG, and PNG formats.
            </p>
            <button
              type="button"
              className="px-4 py-2 rounded-lg bg-navy-800 hover:bg-navy-900 text-white text-xs font-semibold shadow-sm transition-all"
            >
              Browse Local Files
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Image Preview & Details */}
            <div className="relative rounded-2xl overflow-hidden bg-slate-900 flex items-center justify-center max-h-[380px] p-2">
              <img
                src={previewUrl}
                alt="Uploaded radiograph"
                className="max-h-[360px] object-contain rounded-xl"
              />
              <button
                onClick={handleClear}
                disabled={analyzing}
                className="absolute top-4 right-4 p-2 rounded-full bg-slate-900/80 hover:bg-rose-600 text-white transition-colors"
                title="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Quality Assessment Gate Preview */}
            {validationResult && (
              <div className={`p-4 rounded-xl border text-xs ${
                validationResult.valid 
                  ? 'bg-slate-50 border-slate-200' 
                  : 'bg-rose-50 border-rose-200 text-rose-800'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {validationResult.valid ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-rose-600" />
                    )}
                    <span className="font-bold text-navy-900">
                      Technical Quality Assessment: {validationResult.status}
                    </span>
                  </div>
                  <span className="font-mono font-bold text-navy-900">
                    Quality Score: {validationResult.quality_score}/100
                  </span>
                </div>

                {validationResult.metrics && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-600 mt-2 pt-2 border-t border-slate-200/60">
                    <div>Resolution: <b>{validationResult.metrics.width}&times;{validationResult.metrics.height}</b></div>
                    <div>Format: <b>{validationResult.metrics.format}</b></div>
                    <div>Mean Intensity: <b>{validationResult.metrics.mean_intensity}</b></div>
                    <div>Focus Var: <b>{validationResult.metrics.blur_variance}</b></div>
                  </div>
                )}

                {validationResult.warnings?.length > 0 && (
                  <div className="mt-2 text-[11px] text-amber-700 space-y-0.5">
                    {validationResult.warnings.map((w, i) => (
                      <div key={i}>&bull; {w}</div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Action Button */}
            <div className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-2">
              <div className="text-[11px] text-slate-400">
                Loaded: <b>{file?.name}</b> ({Math.round((file?.size || 0) / 1024)} KB)
              </div>
              <div className="flex space-x-3 w-full sm:w-auto">
                <button
                  onClick={handleClear}
                  disabled={analyzing}
                  className="flex-1 sm:flex-none px-4 py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-100 text-xs font-semibold transition-all disabled:opacity-50"
                >
                  Choose Different Image
                </button>
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing || (validationResult && !validationResult.valid)}
                  className="flex-1 sm:flex-none flex items-center justify-center space-x-2 px-6 py-2.5 rounded-xl bg-teal-700 hover:bg-teal-800 text-white text-xs font-bold shadow-md shadow-teal-500/20 hover:shadow-teal-500/30 transition-all disabled:opacity-50"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>{analysisStep || 'Analyzing...'}</span>
                    </>
                  ) : (
                    <>
                      <span>Run RespiraSeverity Pipeline</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
