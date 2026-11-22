import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import AnalysisPage from './pages/AnalysisPage';
import ResultsPage from './pages/ResultsPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import ClusteringPage from './pages/ClusteringPage';
import PerformancePage from './pages/PerformancePage';
import ReportPage from './pages/ReportPage';

export default function App() {
  const [activePage, setActivePage] = useState('landing');
  const [analysisData, setAnalysisData] = useState(null);
  const [history, setHistory] = useState([]);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    
    // Add to history
    const historyItem = {
      timestamp: data.timestamp,
      filename: data.filename,
      disease: data.result.disease?.predicted_class,
      confidence: data.result.disease?.confidence,
      severity: data.result.severity?.severity,
      cluster: data.result.clustering?.cluster_name,
      quality: data.result.quality,
    };
    setHistory((prev) => [historyItem, ...prev]);

    // Automatically navigate to Results page
    setActivePage('results');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Navigation */}
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        hasAnalysisResult={!!analysisData}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activePage === 'landing' && (
          <LandingPage
            onStartAnalysis={() => setActivePage('analysis')}
            onNavigate={(page) => setActivePage(page)}
          />
        )}

        {activePage === 'dashboard' && (
          <DashboardPage
            onNavigate={(page) => setActivePage(page)}
            history={history}
          />
        )}

        {activePage === 'analysis' && (
          <AnalysisPage onAnalysisComplete={handleAnalysisComplete} />
        )}

        {activePage === 'results' && (
          <ResultsPage
            analysisData={analysisData}
            onNavigate={(page) => setActivePage(page)}
          />
        )}

        {activePage === 'explain' && (
          <ExplainabilityPage
            analysisData={analysisData}
            onNavigate={(page) => setActivePage(page)}
          />
        )}

        {activePage === 'clustering' && (
          <ClusteringPage />
        )}

        {activePage === 'performance' && (
          <PerformancePage />
        )}

        {activePage === 'report' && (
          <ReportPage
            analysisData={analysisData}
            onNavigate={(page) => setActivePage(page)}
          />
        )}
      </main>

      {/* Global Academic Footer */}
      <Footer />
    </div>
  );
}
