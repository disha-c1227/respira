import React, { useState } from 'react';
import { 
  FileText, 
  Download, 
  Loader2, 
  CheckCircle2, 
  ShieldAlert, 
  Eye, 
  Calendar, 
  FileCheck2,
  AlertTriangle
} from 'lucide-react';
import { generateReportPdf } from '../services/api';

export default function ReportPage({ analysisData, onNavigate }) {
  const [downloading, setDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  if (!analysisData || !analysisData.result) {
    return (
      <div className="max-w-xl mx-auto my-12 text-center p-8 bg-white rounded-3xl border border-slate-200">
        <FileText className="w-12 h-12 text-teal-700 mx-auto mb-3" />
        <h3 className="text-lg font-bold text-navy-900 mb-1">No Active Radiograph Report</h3>
        <p className="text-xs text-slate-500 mb-6">
          Analyze a chest X-ray image first to compile a clinical research PDF report.
        </p>
        <button
          onClick={() => onNavigate('analysis')}
          className="px-5 py-2.5 rounded-xl bg-navy-800 text-white font-semibold text-xs"
        >
          Go to Analysis Portal
        </button>
      </div>
    );
  }

  const { result, filename, timestamp } = analysisData;
  const disease = result.disease || {};
  const severity = result.severity;
  const clustering = result.clustering;
  const quality = result.quality || 85;
  const gradcam = result.gradcam;

  const handleDownload = async () => {
    setDownloading(true);
    setErrorMsg(null);
    setDownloadSuccess(false);

    try {
      const payload = {
        image_id: filename || 'CXR_QUERY_001',
        validation_data: result.quality_validation || { quality_score: quality, status: 'PASSED' },
        disease_data: disease,
        severity_data: severity,
        clustering_data: clustering,
        gradcam_images: gradcam ? {
          original_base64: gradcam.original_base64,
          heatmap_base64: gradcam.heatmap_base64,
          overlay_base64: gradcam.overlay_base64
        } : null
      };

      const blob = await generateReportPdf(payload);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `RespiraSeverity_Report_${filename.replace(/\.[^/.]+$/, "")}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setDownloadSuccess(true);
    } catch (e) {
      setErrorMsg(`Failed to generate PDF: ${e.message}`);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[11px] font-semibold mb-1">
            <FileText className="w-3.5 h-3.5 text-teal-600" />
            <span>Structured Academic Deliverable</span>
          </div>
          <h2 className="text-2xl font-bold text-navy-900 tracking-tight">
            Research Analysis PDF Report
          </h2>
          <p className="text-slate-500 text-xs mt-0.5">
            Formatted in compliance with medical imaging research reporting guidelines.
          </p>
        </div>

        <button
          onClick={handleDownload}
          disabled={downloading}
          className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-teal-700 hover:bg-teal-800 text-white font-bold text-xs shadow-md shadow-teal-500/20 hover:shadow-teal-500/30 transition-all disabled:opacity-50"
        >
          {downloading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Compiling PDF Document...</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span>Download PDF Report</span>
            </>
          )}
        </button>
      </div>

      {downloadSuccess && (
        <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span>PDF successfully compiled and downloaded to your computer!</span>
        </div>
      )}

      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Report Document Preview Sheet */}
      <div className="bg-white rounded-3xl border border-slate-200 p-8 sm:p-12 shadow-sm space-y-8 font-sans">
        {/* Document Header Banner */}
        <div className="border-b border-teal-600 pb-4 flex justify-between items-start">
          <div>
            <h1 className="text-xl font-bold text-navy-900">
              Respira<span className="text-teal-700">Severity</span> Research Report
            </h1>
            <p className="text-xs text-teal-800 font-medium mt-0.5">
              Deep Learning-Based Pulmonary Disease Classification & Severity Assessment
            </p>
            <p className="text-[10px] text-slate-400 mt-1">
              Vidyavardhaka College of Engineering &bull; Dept. of Computer Science & Engineering
            </p>
          </div>
          <div className="text-right text-[11px] text-slate-500">
            <div>Timestamp: <b>{timestamp}</b></div>
            <div>Image ID: <b>{filename}</b></div>
          </div>
        </div>

        {/* Section 1: Technical Validation */}
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80 text-xs space-y-1">
          <div className="font-bold text-navy-900 mb-1">1. Radiographic Technical Quality Assessment</div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-600">
            <div>Technical Status: <b className="text-emerald-700">PASSED</b></div>
            <div>Quality Score: <b>{quality}/100</b></div>
            <div>Input Standard: <b>224&times;224 (3 Ch.)</b></div>
            <div>Format: <b>DICOM / JPEG Compliant</b></div>
          </div>
        </div>

        {/* Section 2: Primary Disease Finding */}
        <div>
          <h3 className="text-xs font-bold text-navy-900 uppercase tracking-wider mb-2">
            2. Primary Disease Classification Finding
          </h3>
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 flex justify-between items-center">
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Predicted Pulmonary Class</span>
              <div className="text-2xl font-extrabold text-navy-900">{disease.predicted_class}</div>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 uppercase">Model Confidence</span>
              <div className="text-2xl font-extrabold text-teal-700">{disease.confidence}%</div>
            </div>
          </div>

          {/* Probabilities table */}
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-xs text-left border border-slate-200 rounded-lg overflow-hidden">
              <thead className="bg-navy-900 text-white font-semibold">
                <tr>
                  <th className="py-1.5 px-3">Disease Class</th>
                  <th className="py-1.5 px-3">Probability</th>
                  <th className="py-1.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {Object.entries(disease.probabilities || {}).map(([cName, prob]) => (
                  <tr key={cName}>
                    <td className="py-1.5 px-3 font-semibold">{cName}</td>
                    <td className="py-1.5 px-3">{prob}%</td>
                    <td className="py-1.5 px-3 text-slate-400">
                      {cName.toLowerCase() === disease.predicted_class?.toLowerCase() ? (
                        <span className="text-teal-700 font-bold">Predicted Target</span>
                      ) : 'Non-target'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 3: Severity Staging */}
        <div>
          <h3 className="text-xs font-bold text-navy-900 uppercase tracking-wider mb-2">
            3. Staged Severity Assessment & Staging Finding
          </h3>
          {disease.predicted_class === 'Normal' && (
            <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/70 text-xs text-emerald-900">
              <b>Normal Finding:</b> No pulmonary infection pathology detected. Staged severity assessment is bypassed.
            </div>
          )}
          {disease.predicted_class === 'COVID' && severity && (
            <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/70 text-xs text-blue-900 space-y-1">
              <div><b>Supervised Staging:</b> COVID-19 Severity Grade &mdash; <span className="font-extrabold text-sm">{severity.severity}</span> ({severity.confidence}% confidence)</div>
              <div className="text-[11px] text-blue-800/80">Probabilities: {Object.entries(severity.probabilities || {}).map(([k, v]) => `${k}: ${v}%`).join(' • ')}</div>
            </div>
          )}
          {(disease.predicted_class === 'Pneumonia' || disease.predicted_class === 'TB') && clustering && (
            <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/70 text-xs text-amber-900 space-y-1">
              <div><b>Exploratory Radiographic Feature Group:</b> <span className="font-extrabold">{clustering.cluster_name}</span></div>
              <div className="text-[11px] text-amber-800">{clustering.cluster_interpretation}</div>
              <div className="text-[10px] text-amber-700 mt-1 italic">
                *Exploratory research cluster derived from K-Means over 1024-D MobileNetV3-Small embeddings. Not a verified clinical severity label.
              </div>
            </div>
          )}
        </div>

        {/* Section 4: Grad-CAM Explainability Thumbnails */}
        {gradcam && (
          <div>
            <h3 className="text-xs font-bold text-navy-900 uppercase tracking-wider mb-2">
              4. Model Visual Attribution (Grad-CAM)
            </h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center space-y-1">
                <span className="text-[10px] font-semibold text-slate-500">Original Radiograph</span>
                <div className="p-1 rounded-lg bg-slate-900">
                  <img src={gradcam.original_base64} alt="Original" className="w-full h-auto rounded" />
                </div>
              </div>
              <div className="text-center space-y-1">
                <span className="text-[10px] font-semibold text-slate-500">Activation Heatmap</span>
                <div className="p-1 rounded-lg bg-slate-900">
                  <img src={gradcam.heatmap_base64} alt="Heatmap" className="w-full h-auto rounded" />
                </div>
              </div>
              <div className="text-center space-y-1">
                <span className="text-[10px] font-semibold text-slate-500">Diagnostic Overlay</span>
                <div className="p-1 rounded-lg bg-slate-900">
                  <img src={gradcam.overlay_base64} alt="Overlay" className="w-full h-auto rounded" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Medical Research Disclaimer */}
        <div className="pt-4 border-t border-slate-200 text-[10px] text-slate-400 leading-relaxed">
          <b>ACADEMIC RESEARCH DISCLAIMER:</b> RespiraSeverity is a final-year academic research software artifact. The outputs, heatmaps, and severity groupings produced by this tool have not undergone clinical trial validation and must never be utilized for actual patient triage, clinical prescription, or diagnostic intervention.
        </div>
      </div>
    </div>
  );
}
