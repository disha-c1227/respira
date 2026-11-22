import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  LayoutDashboard, 
  UploadCloud, 
  FileCheck2, 
  Layers, 
  BarChart3, 
  FileText, 
  Home,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { fetchHealth } from '../services/api';

export default function Navbar({ activePage, setActivePage, hasAnalysisResult }) {
  const [serverOnline, setServerOnline] = useState(false);
  const [device, setDevice] = useState('');

  useEffect(() => {
    async function check() {
      try {
        const h = await fetchHealth();
        setServerOnline(h.status === 'HEALTHY');
        setDevice(h.device || 'CPU');
      } catch (e) {
        setServerOnline(false);
      }
    }
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'landing', label: 'Home', icon: Home },
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analysis', label: 'X-Ray Analysis', icon: UploadCloud },
    { id: 'results', label: 'Results', icon: FileCheck2, badge: hasAnalysisResult ? 'Active' : null },
    { id: 'explain', label: 'Grad-CAM', icon: Activity },
    { id: 'clustering', label: 'Clustering', icon: Layers },
    { id: 'performance', label: 'Model Metrics', icon: BarChart3 },
    { id: 'report', label: 'PDF Report', icon: FileText },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div 
            onClick={() => setActivePage('landing')} 
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-navy-800 to-teal-700 flex items-center justify-center text-white shadow-md shadow-teal-500/10 group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-navy-900">Respira<span className="text-teal-700">Severity</span></span>
                <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-teal-50 text-teal-800 border border-teal-200">Research v2.0</span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">Pulmonary Disease & Staged Severity Assessment</p>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`relative flex items-center space-x-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive 
                      ? 'bg-navy-800 text-white shadow-sm' 
                      : 'text-slate-600 hover:text-navy-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="ml-1 w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Status Badge */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border bg-slate-50 border-slate-200">
              {serverOnline ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-slate-700">API Live <span className="text-[10px] text-slate-400">({device})</span></span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                  <span className="text-rose-600 font-medium">Offline</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile nav bar */}
      <div className="md:hidden flex overflow-x-auto py-2 px-3 border-t border-slate-100 space-x-1 bg-slate-50">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              className={`flex-shrink-0 flex items-center space-x-1 px-2.5 py-1.5 rounded-md text-xs font-medium ${
                isActive ? 'bg-navy-800 text-white' : 'text-slate-600 bg-white border border-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
}
