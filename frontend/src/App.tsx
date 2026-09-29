import { useState, useEffect, lazy, Suspense, useCallback, Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

import { Sidebar } from './components/Sidebar';
import { TopNavbar } from './components/TopNavbar';
import { DashboardOverview } from './components/DashboardOverview';
import { WeatherChatbot } from './components/WeatherChatbot';
import type { ThreatObject, IndiaRegionId } from './types/weather';

import HistoricalAnalysis from './components/HistoricalAnalysis';

// Lazy loading tab components for code splitting & performance
const LiveRiskMap = lazy(() => import('./components/LiveRiskMap').then(m => ({ default: m.LiveRiskMap })));
const CycloneTracker = lazy(() => import('./components/CycloneTracker').then(m => ({ default: m.CycloneTracker })));
const LocalityExplorer = lazy(() => import('./components/LocalityExplorer').then(m => ({ default: m.LocalityExplorer })));
const AiModelHub = lazy(() => import('./components/AiModelHub').then(m => ({ default: m.AiModelHub })));
const LocationRisk = lazy(() => import('./components/LocationRisk').then(m => ({ default: m.LocationRisk })));
const EventDetail = lazy(() => import('./components/EventDetail').then(m => ({ default: m.EventDetail })));
const AlertCenter = lazy(() => import('./components/AlertCenter').then(m => ({ default: m.AlertCenter })));
const FarmerAdvisory = lazy(() => import('./components/FarmerAdvisory').then(m => ({ default: m.FarmerAdvisory })));
const DisasterDashboard = lazy(() => import('./components/DisasterDashboard').then(m => ({ default: m.DisasterDashboard })));
const AdminPanel = lazy(() => import('./components/AdminPanel').then(m => ({ default: m.AdminPanel })));
const ApiExplorer = lazy(() => import('./components/ApiExplorer').then(m => ({ default: m.ApiExplorer })));
const HowItWorks = lazy(() => import('./components/HowItWorks').then(m => ({ default: m.HowItWorks })));
const AuthPage = lazy(() => import('./components/AuthPage').then(m => ({ default: m.AuthPage })));
const SettingsPanel = lazy(() => import('./components/SettingsPanel').then(m => ({ default: m.SettingsPanel })));

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("StormTrace Tab Component Error Caught:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 max-w-2xl mx-auto my-12 bg-white dark:bg-[#0f1628] rounded-2xl border border-red-200 dark:border-red-900/40 text-center space-y-4 shadow-xl">
          <div className="h-12 w-12 rounded-2xl bg-red-100 dark:bg-red-950/50 border border-red-300 dark:border-red-800 text-red-600 dark:text-red-400 flex items-center justify-center mx-auto text-xl font-bold">
            ⚠️
          </div>
          <h3 className="text-lg font-black text-slate-900 dark:text-slate-100">Module Execution Error</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
            {this.state.error?.message || "An unexpected error occurred in this view."}
          </p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, error: undefined })}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs shadow-md transition-all"
          >
            Reload Module View
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Premium loading spinner
const LoadingFallback = () => (
  <div className="h-96 flex flex-col items-center justify-center space-y-4 text-slate-400">
    <div className="relative">
      <div className="h-12 w-12 border-3 border-blue-600/30 rounded-full"></div>
      <div className="h-12 w-12 border-3 border-blue-600 border-t-transparent rounded-full animate-spin absolute inset-0"></div>
    </div>
    <span className="text-xs font-mono font-semibold animate-pulse">Loading StormTrace Module...</span>
  </div>
);


export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [selectedRegion, setSelectedRegion] = useState<IndiaRegionId>('all');
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [selectedThreat, setSelectedThreat] = useState<ThreatObject | null>(null);
  const [topSearchQuery, setTopSearchQuery] = useState<string>('');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  // Sync data-theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const handleSelectThreatFromMap = useCallback((threat: ThreatObject) => {
    setSelectedThreat(threat);
  }, []);

  const handleTopSearch = useCallback((query: string) => {
    setTopSearchQuery(query);
    setActiveTab('locality');
  }, []);

  const handleNavigate = useCallback((tab: string) => {
    setActiveTab(tab);
    setIsMobileMenuOpen(false); // Close mobile menu on navigation
  }, []);

  // Render active tab content
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
      case 'landing':
      case 'overview':
        return (
          <DashboardOverview
            selectedRegion={selectedRegion}
            onNavigate={handleNavigate}
            onSelectThreat={handleSelectThreatFromMap}
          />
        );
      case 'map':
      case 'live-map':
      case 'gis':
        return (
          <LiveRiskMap
            selectedRegion={selectedRegion}
            onSelectThreat={handleSelectThreatFromMap}
          />
        );
      case 'cyclone':
      case 'cyclone-tracker':
      case 'windy':
        return <CycloneTracker />;
      case 'locality':
      case 'search':
        return <LocalityExplorer initialSearchQuery={topSearchQuery} />;
      case 'models':
      case 'evaluation':
      case 'ai-analysis':
      case 'ai-models':
      case 'model-evaluation':
        return <AiModelHub />;
      case 'location':
      case 'location-risk':
        return (
          <LocationRisk
            initialLocKey="prayagraj"
            onNavigateToEvent={() => setActiveTab('event')}
          />
        );
      case 'event':
      case 'anomaly-tracking':
      case 'tracking':
      case 'event-detail':
        return (
          <EventDetail
            selectedEventId={selectedThreat?.id}
            onNavigateToMap={() => setActiveTab('map')}
          />
        );
      case 'alerts':
      case 'risk-alerts':
      case 'alert-center':
        return <AlertCenter />;
      case 'historical':
      case 'historical-explorer':
      case 'historical-replay':
      case 'historical-analysis':
      case 'history':
      case 'historical_explorer':
      case 'historical_analysis':
        return <HistoricalAnalysis />;
      case 'farmer':
      case 'farmer-advisory':
      case 'advisory':
        return <FarmerAdvisory lang="en" setLang={() => {}} />;
      case 'how-it-works':
      case 'guide':
        return <HowItWorks />;
      case 'disaster':
      case 'operations-room':
      case 'operations':
      case 'disaster-dashboard':
        return <DisasterDashboard />;
      case 'admin':
      case 'data-center':
        return <AdminPanel />;
      case 'settings':
        return <SettingsPanel theme={theme} setTheme={setTheme} />;
      case 'api':
      case 'apis':
      case 'weather-apis':
      case 'api-explorer':
        return <ApiExplorer />;
      case 'auth':
      case 'login':
      case 'signup':
        return <AuthPage onNavigateToTab={handleNavigate} />;
      default:
        return (
          <DashboardOverview
            selectedRegion={selectedRegion}
            onNavigate={handleNavigate}
            onSelectThreat={handleSelectThreatFromMap}
          />
        );
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-slate-100 dark:bg-[#070b16] text-slate-900 dark:text-slate-100 flex font-sans transition-colors duration-300 relative">
      {/* Sidebar */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={handleNavigate}
        isMobileMenuOpen={isMobileMenuOpen}
        setIsMobileMenuOpen={setIsMobileMenuOpen}
      />

      {/* Main Content Shell */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Navbar */}
        <TopNavbar
          selectedRegion={selectedRegion}
          setSelectedRegion={setSelectedRegion}
          theme={theme}
          setTheme={setTheme}
          onSearchSubmit={handleTopSearch}
          onMobileMenuToggle={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          onNavigateToTab={handleNavigate}
        />


        {/* Content */}
        <main 
          id="main-content" 
          className={`flex-1 min-h-0 ${
            activeTab === 'map' || activeTab === 'cyclone' || activeTab === 'cyclone-tracker' || activeTab === 'windy' 
              ? 'p-0 overflow-hidden flex flex-col h-full w-full max-w-none' 
              : 'overflow-y-auto px-3 sm:px-6 pt-4 sm:pt-6 pb-20 sm:pb-12 max-w-7xl w-full mx-auto'
          }`}
        >
          <ErrorBoundary key={activeTab}>
            <Suspense fallback={<LoadingFallback />}>
              {renderContent()}
            </Suspense>
          </ErrorBoundary>
        </main>

        {/* Floating AI Weather Assistant Chatbot */}
        <WeatherChatbot onNavigateToTab={handleNavigate} />
      </div>
    </div>
  );
}

export default App;
