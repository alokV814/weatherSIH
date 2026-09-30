import React, { useState, useEffect, useMemo, lazy, Suspense } from 'react';
import { 
  Activity, 
  MapPin, 
  CloudRain, 
  Thermometer, 
  AlertTriangle, 
  ArrowRight, 
  Layers,
  Sparkles,
  TrendingUp
} from 'lucide-react';

import type { IndiaRegionId, ThreatObject, AlertItem } from '../types/weather';
import { fetchApiAlerts, fetchApiThreatObjects } from '../services/apiService';

// Lazy load to enable proper code splitting with App.tsx
const LiveRiskMap = lazy(() => import('./LiveRiskMap').then(m => ({ default: m.LiveRiskMap })));

interface DashboardOverviewProps {
  selectedRegion: IndiaRegionId;
  onNavigate: (tab: string) => void;
  onSelectThreat: (threat: ThreatObject) => void;
}

// Stat card component for reuse
const StatCard: React.FC<{
  label: string;
  value: string;
  subtext: React.ReactNode;
  icon: React.ElementType;
  iconBg: string;
  iconColor: string;
  accentColor?: string;
}> = ({ label, value, subtext, icon: Icon, iconBg, iconColor, accentColor }) => (
  <div className="storm-card storm-card-hover p-5 flex items-center justify-between stat-shimmer">
    <div>
      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block mb-1">{label}</span>
      <span className={`text-2xl font-black font-mono ${accentColor || 'text-slate-900 dark:text-slate-100'}`}>
        {value}
      </span>
      <div className="mt-1 text-[10px]">{subtext}</div>
    </div>
    <div className={`h-12 w-12 rounded-2xl ${iconBg} border flex items-center justify-center ${iconColor}`}>
      <Icon className="h-6 w-6" />
    </div>
  </div>
);

export const DashboardOverview: React.FC<DashboardOverviewProps> = ({
  selectedRegion,
  onNavigate,
  onSelectThreat,
}) => {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [threatObjects, setThreatObjects] = useState<ThreatObject[]>([]);

  useEffect(() => {
    let isMounted = true;
    async function loadDashboardData() {
      const [alertsRes, threatsRes] = await Promise.all([
        fetchApiAlerts(selectedRegion),
        fetchApiThreatObjects()
      ]);
      if (isMounted) {
        setAlerts(alertsRes.alerts);
        setThreatObjects(threatsRes);
      }
    }
    loadDashboardData();
    return () => { isMounted = false; };
  }, [selectedRegion]);

  const alertStats = useMemo(() => ({
    critical: alerts.filter(a => a.riskLevel === 'critical').length,
    severe: alerts.filter(a => a.riskLevel === 'severe').length,
    total: alerts.length,
  }), [alerts]);

  const threatCount = useMemo(() => threatObjects.length, [threatObjects]);

  const greeting = useMemo(() => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good Morning';
    if (hour < 18) return 'Good Afternoon';
    return 'Good Evening';
  }, []);

  const [liveStats, setLiveStats] = React.useState({
    temp: 'Loading...',
    rain: 'Loading...',
    loading: true
  });

  React.useEffect(() => {
    // Fetch live data for New Delhi (Central reference)
    fetch('https://api.open-meteo.com/v1/forecast?latitude=28.61&longitude=77.23&daily=precipitation_sum&current_weather=true&timezone=Asia/Kolkata')
      .then(res => res.json())
      .then(data => {
        setLiveStats({
          temp: data?.current_weather?.temperature ? `${data.current_weather.temperature}°C` : 'N/A',
          rain: data?.daily?.precipitation_sum?.[0] !== undefined ? `${data.daily.precipitation_sum[0]} mm` : '0 mm',
          loading: false
        });
      })
      .catch(err => {
        console.error('Failed to fetch dashboard live weather:', err);
        setLiveStats({ temp: 'Offline', rain: 'Offline', loading: false });
      });
  }, []);

  return (
    <div className="space-y-6 pb-12">
      {/* Top Welcome Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-[#0f1628] p-6 rounded-2xl border border-slate-200 dark:border-[#1a2540] shadow-sm transition-colors">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1">
            <Sparkles className="h-3.5 w-3.5" />
            <span>AI-Powered Extreme Weather Tracking & Hyperlocal Alerts</span>
          </div>
          <h2 className="text-2xl font-black text-slate-900 dark:text-slate-100 flex items-center gap-3">
            {greeting}, Disaster Officer
            <span className="text-[10px] bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-200 dark:border-indigo-800 tracking-wide font-mono">
              Ensemble Source: NOAA GEFS Seamless (31-member)
            </span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Pan-India weather downscaling and extreme event activity snapshot — {new Date().toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => onNavigate('cyclone')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-red-600 via-orange-600 to-amber-600 hover:from-red-500 hover:to-amber-500 text-white font-black text-xs flex items-center gap-2 transition-all shadow-lg shadow-red-600/30 hover:scale-105"
          >
            <span className="animate-spin text-sm" style={{ animationDuration: '4s' }}>🌀</span>
            Windy Cyclone Tracker
            <span className="bg-white/20 text-white px-1.5 py-0.5 rounded text-[10px] uppercase font-mono">Chinta ki baat?</span>
          </button>
          <button
            onClick={() => onNavigate('map')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs flex items-center gap-2 transition-all shadow-lg shadow-blue-600/20 hover:shadow-blue-500/30"
          >
            <Layers className="h-4 w-4" />
            Open GIS Map
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

      </div>

      {/* Windy Cyclone Tracking Callout Card */}
      <div 
        onClick={() => onNavigate('cyclone')}
        className="p-5 rounded-2xl bg-gradient-to-r from-[#0f172a] via-[#1e1b4b] to-[#311228] border border-red-500/40 shadow-xl cursor-pointer hover:border-red-400 transition-all flex flex-col md:flex-row items-center justify-between gap-4 group"
      >
        <div className="flex items-center gap-4">
          <div className="h-14 w-14 rounded-2xl bg-red-600/20 border border-red-500/50 flex items-center justify-center text-3xl shrink-0 animate-pulse">
            🌀
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-full bg-red-500/30 text-red-300 border border-red-500/40 text-[10px] font-black uppercase">
                Active Cyclone Alert
              </span>
              <span className="text-xs text-slate-400 font-mono">BOB-02 Bay of Bengal</span>
            </div>
            <h3 className="text-base font-black text-white mt-1 group-hover:text-red-300 transition">
              Chinta ki Baat hai 😳? Severe Cyclonic Storm REMAL Multi-Model Track
            </h3>
            <p className="text-xs text-slate-300 mt-0.5">
              Live trajectory, cone of uncertainty, wind knots, pressure hPa popups, and IMD vs UKM vs ECMWF vs StormTrace AI comparison!
            </p>
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-2 bg-red-600 hover:bg-red-500 text-white px-4 py-2.5 rounded-xl font-black text-xs shadow-lg transition">
          <span>Launch Windy Tracker</span>
          <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
        </div>
      </div>


      {/* 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Active Events"
          value={`${threatCount} Tracked`}
          subtext={
            <div className="flex items-center gap-2 text-slate-500">
              <span className="text-red-500 font-bold">Critical: {alertStats.critical}</span> • 
              <span className="text-amber-500 font-bold"> Severe: {alertStats.severe}</span>
            </div>
          }
          icon={Activity}
          iconBg="bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900/40"
          iconColor="text-red-600 dark:text-red-400"
        />
        <StatCard
          label="Areas at Risk"
          value="12 Districts"
          subtext={
            <div className="flex items-center gap-2 text-slate-500">
              <span className="text-red-500 font-bold">High: 5</span> • 
              <span className="text-amber-500 font-bold"> Mod: 5</span> • 
              <span className="text-emerald-500 font-bold"> Low: 2</span>
            </div>
          }
          icon={MapPin}
          iconBg="bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/40"
          iconColor="text-amber-600 dark:text-amber-400"
        />
        <StatCard
          label="Total Precipitation (Live)"
          value={liveStats.loading ? '...' : liveStats.rain}
          subtext={<span className="text-slate-500 font-mono">24h Data (New Delhi Ref)</span>}
          icon={CloudRain}
          iconBg="bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-900/40"
          iconColor="text-blue-600 dark:text-blue-400"
          accentColor="text-blue-600 dark:text-blue-400"
        />
        <StatCard
          label="Current Temperature"
          value={liveStats.loading ? '...' : liveStats.temp}
          subtext={
            <span className="text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1">
              <TrendingUp className="h-3 w-3" />
              Live API Sync
            </span>
          }
          icon={Thermometer}
          iconBg="bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/40"
          iconColor="text-emerald-600 dark:text-emerald-400"
        />
      </div>

      {/* Map + Alerts Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Map Preview */}
        <div className="lg:col-span-8 storm-card overflow-hidden flex flex-col h-[520px]">
          <div className="p-4 border-b border-slate-200 dark:border-[#1a2540] flex items-center justify-between bg-white dark:bg-[#0f1628]">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                Active Weather Events & 3D Downscaled Radar
              </h3>
            </div>
            <button
              onClick={() => onNavigate('map')}
              className="text-xs text-blue-600 dark:text-blue-400 font-bold hover:underline flex items-center gap-1"
            >
              Expand Full Map <ArrowRight className="h-3 w-3" />
            </button>
          </div>
          <div className="flex-1 relative">
            <Suspense fallback={
              <div className="h-full flex items-center justify-center bg-slate-50 dark:bg-[#0a0f1e]">
                <div className="text-xs text-slate-400 font-mono animate-pulse">Loading 3D Map Engine...</div>
              </div>
            }>
              <LiveRiskMap selectedRegion={selectedRegion} onSelectThreat={onSelectThreat} />
            </Suspense>
          </div>
        </div>

        {/* Recent Alerts Feed */}
        <div className="lg:col-span-4 storm-card p-5 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1a2540] pb-3 mb-3">
              <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-red-500" />
                Recent Hazard Alerts
              </h3>
              <button
                onClick={() => onNavigate('alerts')}
                className="text-xs text-blue-600 dark:text-blue-400 font-bold hover:underline"
              >
                View All
              </button>
            </div>

            <div className="space-y-3">
              {alerts.slice(0, 3).map((alert) => (
                <div
                  key={alert.id}
                  onClick={() => onNavigate('alerts')}
                  className="p-3.5 rounded-xl bg-slate-50 dark:bg-[#111827]/60 border border-slate-200 dark:border-[#1e2d48] hover:border-blue-500/50 dark:hover:border-blue-500/40 cursor-pointer transition-all space-y-1 group"
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-[9px] px-2 py-0.5 rounded font-mono font-bold uppercase ${
                      alert.riskLevel === 'critical' ? 'bg-red-100 text-red-700 dark:bg-red-950/50 dark:text-red-300' :
                      'bg-orange-100 text-orange-700 dark:bg-orange-950/50 dark:text-orange-300'
                    }`}>
                      {alert.riskLevel}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">{alert.district}</span>
                  </div>
                  <h4 className="font-bold text-slate-800 dark:text-slate-200 text-xs line-clamp-1 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {alert.title}
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 leading-relaxed">
                    {alert.summary}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => onNavigate('alerts')}
            className="w-full py-2.5 rounded-xl bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1e2d48] text-slate-800 dark:text-slate-200 font-bold text-xs flex items-center justify-center gap-2 transition-all border border-slate-200 dark:border-[#1e2d48]"
          >
            Open Alert Center Dispatch
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
