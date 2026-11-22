import React from 'react';
import { ShieldAlert, Award, FileCode2 } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-16 text-slate-600 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {/* Institutional block */}
          <div>
            <div className="flex items-center space-x-2 text-navy-900 font-semibold mb-2">
              <Award className="w-4 h-4 text-teal-700" />
              <span>Academic Research Project</span>
            </div>
            <p className="text-slate-500 leading-relaxed text-[11px]">
              <b>Department of Computer Science & Engineering</b><br/>
              Vidyavardhaka College of Engineering, Mysuru, Karnataka, India.<br/>
              <span className="text-slate-400">Under the guidance of Prof. Harshith C.</span>
            </p>
          </div>

          {/* Authors */}
          <div>
            <div className="flex items-center space-x-2 text-navy-900 font-semibold mb-2">
              <FileCode2 className="w-4 h-4 text-teal-700" />
              <span>Project Investigators</span>
            </div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Disha C &bull; P R Sri Nischal &bull; Ranjith K S<br/>
              <span className="text-slate-400">RespiraSeverity Framework &bull; Final Year Dissertation</span>
            </p>
          </div>

          {/* Medical Disclaimer */}
          <div className="bg-amber-50/70 border border-amber-200/80 rounded-lg p-3">
            <div className="flex items-center space-x-1.5 text-amber-900 font-semibold mb-1 text-[11px]">
              <ShieldAlert className="w-4 h-4 text-amber-700 flex-shrink-0" />
              <span>Research Use Disclaimer</span>
            </div>
            <p className="text-amber-800/90 text-[10px] leading-relaxed">
              This system is an academic machine learning research prototype. Unsupervised Pneumonia/TB clusters and supervised COVID-19 predictions are not clinical diagnoses.
            </p>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row justify-between items-center text-[11px] text-slate-400">
          <p>&copy; {new Date().getFullYear()} RespiraSeverity Research Group. MobileNetV3-Small Backbone.</p>
          <p className="mt-2 sm:mt-0">FastAPI &bull; PyTorch &bull; React &bull; Tailwind CSS</p>
        </div>
      </div>
    </footer>
  );
}
