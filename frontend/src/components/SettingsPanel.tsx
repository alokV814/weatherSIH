import React, { useState, useEffect } from 'react';
import { Settings, Bell, Shield, Moon, Sun, Globe, CheckCircle2, Trash2, LogOut } from 'lucide-react';

interface SettingsPanelProps {
  theme?: 'light' | 'dark';
  setTheme?: (theme: 'light' | 'dark') => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ theme, setTheme }) => {
  const [activeTab, setActiveTab] = useState('general');
  const [tempUnit, setTempUnit] = useState('Celsius (°C)');
  const [rainUnit, setRainUnit] = useState('Millimeters (mm)');
  const [defaultRegion, setDefaultRegion] = useState('Pan-India Overview');
  const [timeFormat, setTimeFormat] = useState('12-hour (AM/PM)');
  const [mapStyle, setMapStyle] = useState('Dark Topographic (Default)');

  const [notifications, setNotifications] = useState({
    criticalAlerts: true,
    dailySummary: false,
    systemHealth: true,
    farmerAdvisories: false,
  });

  const [savedSuccess, setSavedSuccess] = useState(false);
  const [cacheCleared, setCacheCleared] = useState(false);

  // Load stored settings on mount
  useEffect(() => {
    const stored = localStorage.getItem('STORMTRACE_SETTINGS_V1');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (parsed.tempUnit) setTempUnit(parsed.tempUnit);
        if (parsed.rainUnit) setRainUnit(parsed.rainUnit);
        if (parsed.defaultRegion) setDefaultRegion(parsed.defaultRegion);
        if (parsed.timeFormat) setTimeFormat(parsed.timeFormat);
        if (parsed.mapStyle) setMapStyle(parsed.mapStyle);
        if (parsed.notifications) setNotifications(parsed.notifications);
      } catch (e) {}
    }
  }, []);

  const handleSaveSettings = () => {
    const config = {
      tempUnit,
      rainUnit,
      defaultRegion,
      timeFormat,
      mapStyle,
      notifications,
    };
    localStorage.setItem('STORMTRACE_SETTINGS_V1', JSON.stringify(config));
    window.dispatchEvent(new Event('settings-change'));
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  const handleClearCache = () => {
    localStorage.removeItem('STORMTRACE_ALERTS_DB_V1');
    localStorage.removeItem('STORMTRACE_DATASETS_DB_V1');
    localStorage.removeItem('STORMTRACE_SETTINGS_V1');
    setCacheCleared(true);
    setTimeout(() => setCacheCleared(false), 3000);
  };

  const handleSignOut = () => {
    localStorage.removeItem('STORMTRACE_AUTH_USER');
    window.dispatchEvent(new Event('auth-change'));
  };

  const currentIsDark = theme === 'dark' || (!theme && typeof document !== 'undefined' && document.documentElement.classList.contains('dark'));

  return (
    <div className="h-full overflow-y-auto bg-slate-50 dark:bg-[#070b16] p-4 md:p-6 lg:p-8 custom-scrollbar">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header Section */}
        <div className="relative rounded-2xl md:rounded-[32px] overflow-hidden bg-gradient-to-br from-slate-900 to-[#0f1628] p-6 md:p-10 shadow-2xl border border-[#1e2d48]">
          <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-3">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700/50 text-slate-300 text-xs font-semibold uppercase tracking-wider backdrop-blur-md">
                <Settings className="h-3.5 w-3.5" />
                SYSTEM PREFERENCES
              </div>
              <h1 className="text-3xl md:text-5xl font-black text-white tracking-tight leading-[1.1]">
                Settings &amp; Configuration
              </h1>
              <p className="text-slate-400 text-sm md:text-base max-w-xl leading-relaxed font-medium">
                Customize your StormTrace AI experience. Adjust display units, notification preferences, and application appearance.
              </p>
            </div>
          </div>
          <div className="absolute top-0 right-0 p-12 opacity-10 pointer-events-none transform translate-x-1/4 -translate-y-1/4">
            <Settings className="w-64 h-64 md:w-96 md:h-96 text-white animate-[spin_60s_linear_infinite]" />
          </div>
        </div>

        {/* Feedback Banners */}
        {savedSuccess && (
          <div className="p-4 rounded-xl bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 font-bold text-xs flex items-center justify-between animate-in fade-in duration-300">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Settings and preferences saved successfully!</span>
            </div>
          </div>
        )}
        {cacheCleared && (
          <div className="p-4 rounded-xl bg-amber-950/80 border border-amber-500/50 text-amber-300 font-bold text-xs flex items-center justify-between animate-in fade-in duration-300">
            <div className="flex items-center gap-2">
              <Trash2 className="h-4 w-4 text-amber-400" />
              <span>Local cache and persistent alerts database cleared!</span>
            </div>
          </div>
        )}

        {/* Content Section */}
        <div className="bg-white dark:bg-[#0f1628] rounded-2xl border border-slate-200 dark:border-[#1e2d48] shadow-sm overflow-hidden">
          
          {/* Tabs */}
          <div className="flex overflow-x-auto border-b border-slate-200 dark:border-[#1e2d48] custom-scrollbar">
            {['general', 'notifications', 'appearance', 'security'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-4 text-sm font-bold capitalize whitespace-nowrap transition-colors ${
                  activeTab === tab 
                    ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50/50 dark:bg-blue-900/10' 
                    : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-6 md:p-8 space-y-8">
            
            {activeTab === 'general' && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h3 className="text-lg font-black text-slate-900 dark:text-white flex items-center gap-2">
                  <Globe className="h-5 w-5 text-blue-500" /> General Preferences
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Temperature Unit</label>
                    <select 
                      value={tempUnit}
                      onChange={(e) => setTempUnit(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-[#111827] border border-slate-200 dark:border-[#1e2d48] rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 cursor-pointer"
                    >
                      <option>Celsius (°C)</option>
                      <option>Fahrenheit (°F)</option>
                      <option>Kelvin (K)</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Rainfall Unit</label>
                    <select 
                      value={rainUnit}
                      onChange={(e) => setRainUnit(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-[#111827] border border-slate-200 dark:border-[#1e2d48] rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 cursor-pointer"
                    >
                      <option>Millimeters (mm)</option>
                      <option>Inches (in)</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Default Region</label>
                    <select 
                      value={defaultRegion}
                      onChange={(e) => setDefaultRegion(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-[#111827] border border-slate-200 dark:border-[#1e2d48] rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 cursor-pointer"
                    >
                      <option>Pan-India Overview</option>
                      <option>Mumbai Coast</option>
                      <option>Uttar Pradesh (Ganges)</option>
                      <option>Kerala / Western Ghats</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Time Format</label>
                    <select 
                      value={timeFormat}
                      onChange={(e) => setTimeFormat(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-[#111827] border border-slate-200 dark:border-[#1e2d48] rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 cursor-pointer"
                    >
                      <option>12-hour (AM/PM)</option>
                      <option>24-hour</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h3 className="text-lg font-black text-slate-900 dark:text-white flex items-center gap-2">
                  <Bell className="h-5 w-5 text-red-500" /> Alert &amp; Notification Settings
                </h3>
                
                <div className="space-y-4">
                  {[
                    { key: 'criticalAlerts', title: "Critical Weather Alerts", desc: "Receive immediate push notifications for extreme weather events (Red/Orange alerts)." },
                    { key: 'dailySummary', title: "Daily Forecast Summary", desc: "Get a morning briefing on expected conditions in your region." },
                    { key: 'systemHealth', title: "System Health & API Status", desc: "Notify when background models complete or if API endpoints fail." },
                    { key: 'farmerAdvisories', title: "Farmer Advisories", desc: "Receive automated crop risk recommendations." }
                  ].map((item) => {
                    const isChecked = notifications[item.key as keyof typeof notifications];
                    return (
                      <div key={item.key} className="flex items-start justify-between p-4 rounded-xl border border-slate-100 dark:border-[#1e2d48] hover:bg-slate-50 dark:hover:bg-[#111827] transition-colors">
                        <div className="space-y-1 pr-4">
                          <h4 className="font-bold text-sm text-slate-900 dark:text-white">{item.title}</h4>
                          <p className="text-xs text-slate-500 dark:text-slate-400">{item.desc}</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer shrink-0 mt-1">
                          <input 
                            type="checkbox" 
                            className="sr-only peer" 
                            checked={isChecked}
                            onChange={(e) => setNotifications(prev => ({ ...prev, [item.key]: e.target.checked }))} 
                          />
                          <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {activeTab === 'appearance' && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h3 className="text-lg font-black text-slate-900 dark:text-white flex items-center gap-2">
                  <Sun className="h-5 w-5 text-amber-500" /> Appearance &amp; Theme
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div 
                    onClick={() => {
                      if (setTheme) setTheme('dark');
                      document.documentElement.classList.add('dark');
                      document.documentElement.setAttribute('data-theme', 'dark');
                    }}
                    className={`p-4 rounded-xl border-2 transition-all cursor-pointer relative overflow-hidden flex flex-col items-center gap-3 ${
                      currentIsDark
                        ? 'border-blue-500 bg-slate-900 text-white shadow-lg'
                        : 'border-slate-200 dark:border-[#1e2d48] bg-slate-50 text-slate-900 hover:border-blue-400'
                    }`}
                  >
                    <Moon className="h-8 w-8 text-blue-400" />
                    <span className="font-bold text-sm">Dark Theme {currentIsDark ? '(Active)' : ''}</span>
                    {currentIsDark && (
                      <div className="absolute top-2 right-2 h-3 w-3 rounded-full bg-blue-500"></div>
                    )}
                  </div>

                  <div 
                    onClick={() => {
                      if (setTheme) setTheme('light');
                      document.documentElement.classList.remove('dark');
                      document.documentElement.setAttribute('data-theme', 'light');
                    }}
                    className={`p-4 rounded-xl border-2 transition-all cursor-pointer flex flex-col items-center gap-3 ${
                      !currentIsDark
                        ? 'border-blue-500 bg-blue-50 dark:bg-slate-800 text-slate-900 dark:text-white shadow-lg'
                        : 'border-slate-200 dark:border-[#1e2d48] bg-slate-50 text-slate-900 hover:border-amber-400'
                    }`}
                  >
                    <Sun className="h-8 w-8 text-amber-500" />
                    <span className="font-bold text-sm dark:text-slate-300">Light Theme {!currentIsDark ? '(Active)' : ''}</span>
                    {!currentIsDark && (
                      <div className="absolute top-2 right-2 h-3 w-3 rounded-full bg-amber-500"></div>
                    )}
                  </div>
                </div>

                <div className="space-y-2 pt-4">
                  <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Map Style</label>
                  <select 
                    value={mapStyle}
                    onChange={(e) => setMapStyle(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-[#111827] border border-slate-200 dark:border-[#1e2d48] rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 cursor-pointer"
                  >
                    <option>Dark Topographic (Default)</option>
                    <option>Satellite Hybrid</option>
                    <option>Street View</option>
                    <option>Monochrome</option>
                  </select>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h3 className="text-lg font-black text-slate-900 dark:text-white flex items-center gap-2">
                  <Shield className="h-5 w-5 text-emerald-500" /> Security &amp; Session
                </h3>
                
                <div className="space-y-4">
                  <div className="p-4 rounded-xl border border-slate-200 dark:border-[#1e2d48] bg-slate-50 dark:bg-[#111827] flex items-center justify-between">
                    <div>
                      <h4 className="font-bold text-sm text-slate-900 dark:text-white">Current Session</h4>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Logged in via Web Browser • Active Status: Online</p>
                    </div>
                    <button 
                      onClick={handleSignOut}
                      className="px-4 py-2 bg-slate-200 dark:bg-slate-800 hover:bg-red-600 hover:text-white text-slate-800 dark:text-white rounded-lg text-xs font-bold transition-all flex items-center gap-1.5"
                    >
                      <LogOut className="h-3.5 w-3.5" />
                      Sign Out
                    </button>
                  </div>
                  
                  <div className="p-4 rounded-xl border border-red-200 dark:border-red-900/30 bg-red-50/50 dark:bg-red-950/10 flex items-center justify-between">
                    <div>
                      <h4 className="font-bold text-sm text-red-600 dark:text-red-400">Clear Application Data</h4>
                      <p className="text-xs text-red-500/80 dark:text-red-400/70 mt-1">Remove all cached models, local alerts, and preferences.</p>
                    </div>
                    <button 
                      onClick={handleClearCache}
                      className="px-4 py-2 bg-red-100 dark:bg-red-900/50 hover:bg-red-600 hover:text-white text-red-700 dark:text-red-300 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Clear Cache
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Save Button */}
            <div className="pt-8 flex justify-end">
              <button 
                onClick={handleSaveSettings}
                className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-sm shadow-md hover:shadow-lg transition-all flex items-center gap-2"
              >
                <Settings className="h-4 w-4" />
                Save Changes
              </button>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
};

