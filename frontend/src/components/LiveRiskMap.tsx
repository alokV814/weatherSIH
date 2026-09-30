import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { 
  Layers, 
  Play, 
  Pause, 
  Maximize2, 
  Minimize2, 
  Sliders, 
  RefreshCw,
  Globe,
  Mountain,
  Eye,
  EyeOff,
  RotateCcw,
  Compass,
} from 'lucide-react';
import type { MapLayerId, ThreatObject, GridCell5km, IndiaRegionId } from '../types/weather';
import { INDIA_REGION_PRESETS } from '../data/mockData';
import { fetchApiThreatObjects, fetchApiRiskGrid } from '../services/apiService';
import { API_CONFIG, getOpenWeatherTileUrl } from '../config/apiConfig';

interface LiveRiskMapProps {
  selectedRegion?: IndiaRegionId;
  onSelectThreat?: (threat: ThreatObject) => void;
}

const RISK_COLORS: Record<string, string> = {
  critical: '#ef4444',
  severe: '#f97316',
  moderate: '#f59e0b',
  low: '#10b981',
};

const RISK_GLOW: Record<string, string> = {
  critical: 'rgba(239, 68, 68, 0.4)',
  severe: 'rgba(249, 115, 22, 0.35)',
  moderate: 'rgba(245, 158, 11, 0.3)',
  low: 'rgba(16, 185, 129, 0.25)',
};

// Pan-India Isohyets GeoJSON features representing downscaled rainfall zones across India
const PAN_INDIA_RAINFALL_FEATURES = [
  {
    name: 'Supaul Kosi Catchment Heavy Downpour',
    district: 'Supaul',
    state: 'Bihar',
    rainMm: 165.0,
    efiPercentile: 98.8,
    probGt50: 95,
    riskLevel: 'critical',
    coords: [
      [86.4, 25.8], [86.9, 25.8], [87.1, 26.4], [86.7, 26.6], [86.3, 26.2], [86.4, 25.8]
    ]
  },
  {
    name: 'Mumbai Suburban Urban Cloudburst Cell',
    district: 'Mumbai Suburban',
    state: 'Maharashtra',
    rainMm: 135.0,
    efiPercentile: 99.2,
    probGt50: 97,
    riskLevel: 'critical',
    coords: [
      [72.7, 18.9], [73.1, 18.9], [73.2, 19.3], [72.8, 19.4], [72.6, 19.1], [72.7, 18.9]
    ]
  },
  {
    name: 'Wayanad Orographic Monsoon Downpour',
    district: 'Wayanad',
    state: 'Kerala',
    rainMm: 185.0,
    efiPercentile: 99.6,
    probGt50: 98,
    riskLevel: 'critical',
    coords: [
      [75.9, 11.4], [76.4, 11.4], [76.5, 11.9], [76.0, 12.0], [75.8, 11.6], [75.9, 11.4]
    ]
  },
  {
    name: 'Brahmaputra Middle Valley Inundation',
    district: 'Kamrup Metropolitan',
    state: 'Assam',
    rainMm: 110.0,
    efiPercentile: 96.5,
    probGt50: 91,
    riskLevel: 'severe',
    coords: [
      [91.4, 25.9], [92.1, 25.9], [92.3, 26.4], [91.7, 26.6], [91.3, 26.2], [91.4, 25.9]
    ]
  },
  {
    name: 'Mahanadi Coastal Delta Storm Rainfall',
    district: 'Cuttack',
    state: 'Odisha',
    rainMm: 145.0,
    efiPercentile: 98.1,
    probGt50: 94,
    riskLevel: 'critical',
    coords: [
      [85.6, 20.1], [86.4, 20.1], [86.5, 20.7], [85.9, 20.8], [85.5, 20.4], [85.6, 20.1]
    ]
  },
  {
    name: 'Prayagraj Sangam Confluence Flash Cell',
    district: 'Prayagraj',
    state: 'Uttar Pradesh',
    rainMm: 118.4,
    efiPercentile: 97.4,
    probGt50: 92,
    riskLevel: 'critical',
    coords: [
      [81.6, 25.2], [82.1, 25.2], [82.2, 25.7], [81.7, 25.8], [81.5, 25.4], [81.6, 25.2]
    ]
  },
  {
    name: 'Delhi-NCR & Yamuna Catchment Rain',
    district: 'New Delhi',
    state: 'Delhi',
    rainMm: 95.0,
    efiPercentile: 95.2,
    probGt50: 84,
    riskLevel: 'severe',
    coords: [
      [76.9, 28.3], [77.5, 28.3], [77.6, 28.9], [77.0, 29.0], [76.8, 28.5], [76.9, 28.3]
    ]
  },
  {
    name: 'Chamoli Alaknanda Himalayan Surge',
    district: 'Chamoli',
    state: 'Uttarakhand',
    rainMm: 155.0,
    efiPercentile: 99.1,
    probGt50: 96,
    riskLevel: 'critical',
    coords: [
      [79.1, 30.1], [79.8, 30.1], [79.9, 30.7], [79.3, 30.8], [79.0, 30.4], [79.1, 30.1]
    ]
  },
  {
    name: 'South Peninsular Coastal Surge Rain',
    district: 'Chennai',
    state: 'Tamil Nadu',
    rainMm: 88.0,
    efiPercentile: 94.0,
    probGt50: 78,
    riskLevel: 'moderate',
    coords: [
      [80.0, 12.8], [80.5, 12.8], [80.6, 13.3], [80.1, 13.4], [79.9, 13.0], [80.0, 12.8]
    ]
  },
  {
    name: 'Sundarbans Bay Convective Cell',
    district: 'South 24 Parganas',
    state: 'West Bengal',
    rainMm: 130.0,
    efiPercentile: 97.8,
    probGt50: 93,
    riskLevel: 'severe',
    coords: [
      [88.1, 21.6], [88.8, 21.6], [88.9, 22.2], [88.3, 22.3], [88.0, 21.9], [88.1, 21.6]
    ]
  },
  {
    name: 'South Gujarat Surat Coastal Belt',
    district: 'Surat',
    state: 'Gujarat',
    rainMm: 105.0,
    efiPercentile: 96.0,
    probGt50: 88,
    riskLevel: 'severe',
    coords: [
      [72.6, 21.0], [73.2, 21.0], [73.3, 21.5], [72.7, 21.6], [72.5, 21.2], [72.6, 21.0]
    ]
  },
  {
    name: 'Western Ghats Orographic Belt',
    district: 'Ratnagiri',
    state: 'Maharashtra',
    rainMm: 175.0,
    efiPercentile: 99.4,
    probGt50: 97,
    riskLevel: 'critical',
    coords: [
      [73.1, 16.8], [73.7, 16.8], [73.8, 17.4], [73.2, 17.5], [73.0, 17.0], [73.1, 16.8]
    ]
  },
  {
    name: 'Rohilkhand Ganges Upper Basin Rain',
    district: 'Shahjahanpur',
    state: 'Uttar Pradesh',
    rainMm: 125.0,
    efiPercentile: 97.0,
    probGt50: 90,
    riskLevel: 'severe',
    coords: [
      [79.6, 27.6], [80.2, 27.6], [80.3, 28.2], [79.7, 28.3], [79.5, 27.8], [79.6, 27.6]
    ]
  }
];

const FREE_DARK_STYLE: mapboxgl.Style = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
      ],
      tileSize: 256,
      maxzoom: 20,
      attribution: '© OpenStreetMap contributors © CARTO'
    }
  },
  layers: [
    {
      id: 'carto-dark-base',
      type: 'raster',
      source: 'carto-dark',
      minzoom: 0,
      maxzoom: 20
    }
  ]
};

const FREE_SATELLITE_STYLE: mapboxgl.Style = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      maxzoom: 18,
      attribution: '© Esri © OpenStreetMap contributors'
    },
    'esri-satellite-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      maxzoom: 18,
      attribution: '© Esri'
    }
  },
  layers: [
    {
      id: 'esri-satellite-base',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 18
    },
    {
      id: 'esri-satellite-ref-layer',
      type: 'raster',
      source: 'esri-satellite-ref',
      minzoom: 0,
      maxzoom: 18
    }
  ]
};

function getMapStyleSpec(styleType: 'dark' | 'satellite'): string | mapboxgl.Style {
  const token = API_CONFIG.mapboxPublicToken;
  if (token && token.startsWith('pk.')) {
    return styleType === 'dark' 
      ? 'mapbox://styles/mapbox/dark-v11' 
      : 'mapbox://styles/mapbox/satellite-streets-v12';
  }
  return styleType === 'dark' ? FREE_DARK_STYLE : FREE_SATELLITE_STYLE;
}

export const LiveRiskMap: React.FC<LiveRiskMapProps> = ({ selectedRegion = 'all', onSelectThreat }) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const markersRef = useRef<mapboxgl.Marker[]>([]);
  const popupsRef = useRef<mapboxgl.Popup[]>([]);

  const [activeLayers, setActiveLayers] = useState<Record<MapLayerId, boolean>>({
    rainfall_forecast: false,
    rainfall_anomaly: false,
    extreme_probability: false,
    threat_footprint: false,
    trajectory: false,
    risk_grid_5km: false,
    admin_boundaries: true,
    vulnerability: false,
    wind_extremes: false,
  });


  const [currentRegion, setCurrentRegion] = useState<IndiaRegionId>(selectedRegion);
  const [selectedTimeStep, setSelectedTimeStep] = useState<number>(12);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [layerOpacity, setLayerOpacity] = useState<number>(0.80);
  const [riskGrid, setRiskGrid] = useState<GridCell5km[]>([]);
  const [selectedCell, setSelectedCell] = useState<GridCell5km | null>(null);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [is3DEnabled, setIs3DEnabled] = useState<boolean>(true);
  const [showLayerPanel, setShowLayerPanel] = useState<boolean>(false);
  const [showCellPanel, setShowCellPanel] = useState<boolean>(false);
  const [threatObjects, setThreatObjects] = useState<ThreatObject[]>([]);
  const [mapStyle, setMapStyle] = useState<'dark' | 'satellite'>('dark');

  const [liveFeatures, setLiveFeatures] = useState<any[]>(PAN_INDIA_RAINFALL_FEATURES);
  const [lastSyncTime, setLastSyncTime] = useState<string>('');
  const [isSyncing, setIsSyncing] = useState<boolean>(false);

  const syncLiveTelemetry = useCallback(async (stepHour: number = 0) => {
    setIsSyncing(true);
    try {
      const centroids = PAN_INDIA_RAINFALL_FEATURES.map(f => {
        const sum = f.coords.reduce((acc, c) => [acc[0] + c[0], acc[1] + c[1]], [0, 0]);
        return {
          lat: Number((sum[1] / f.coords.length).toFixed(2)),
          lng: Number((sum[0] / f.coords.length).toFixed(2)),
        };
      });

      const lats = centroids.map(c => c.lat).join(',');
      const lons = centroids.map(c => c.lng).join(',');

      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lats}&longitude=${lons}&daily=precipitation_sum,precipitation_probability_max,wind_speed_10m_max&hourly=precipitation,precipitation_probability&current=precipitation,rain,showers,weather_code,wind_speed_10m&timezone=Asia/Kolkata`;

      const res = await fetch(url);
      if (!res.ok) throw new Error('Open-Meteo HTTP error');
      const rawData = await res.json();
      const dataList = Array.isArray(rawData) ? rawData : [rawData];

      const updated = PAN_INDIA_RAINFALL_FEATURES.map((base, idx) => {
        const weather = dataList[idx] || {};
        let rain = 0.0;
        let prob = 0;

        if (stepHour <= 0) {
          rain = weather.daily?.precipitation_sum?.[0] ?? weather.current?.precipitation ?? 0.0;
          prob = weather.daily?.precipitation_probability_max?.[0] ?? 0;
        } else {
          const hourlyRainArr = weather.hourly?.precipitation || [];
          const hourlyProbArr = weather.hourly?.precipitation_probability || [];
          const hIdx = Math.min(stepHour, hourlyRainArr.length - 1);
          rain = hourlyRainArr[hIdx] ?? 0.0;
          prob = hourlyProbArr[hIdx] ?? 0;
        }

        rain = Math.round(rain * 10) / 10;
        prob = Math.min(99, Math.max(0, Math.round(prob)));

        let riskLevel: 'low' | 'moderate' | 'severe' | 'critical' = 'low';
        if (rain >= 100 || (prob > 90 && rain > 40)) {
          riskLevel = 'critical';
        } else if (rain >= 35 || (prob > 70 && rain > 15)) {
          riskLevel = 'severe';
        } else if (rain >= 5 || prob > 30) {
          riskLevel = 'moderate';
        } else {
          riskLevel = 'low';
        }

        const efiPercentile = Math.min(99.9, Math.max(10.0, Math.round((prob * 0.7) + (rain > 10 ? 25 : 5))));

        return {
          ...base,
          rainMm: rain,
          probGt50: prob,
          efiPercentile,
          riskLevel,
        };
      });

      setLiveFeatures(updated);
      setLastSyncTime(new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    } catch (err) {
      console.warn('Live map telemetry sync failed, using baseline values:', err);
    } finally {
      setIsSyncing(false);
    }
  }, []);

  useEffect(() => {
    syncLiveTelemetry(selectedTimeStep);
  }, [selectedTimeStep, syncLiveTelemetry]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const isohyetFeatures = liveFeatures.map((item, idx) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Polygon' as const,
        coordinates: [item.coords],
      },
      properties: {
        id: `ISO-${idx + 1}`,
        name: item.name,
        district: item.district,
        state: item.state,
        rainMm: item.rainMm,
        efiPercentile: item.efiPercentile,
        probGt50: item.probGt50,
        riskLevel: item.riskLevel,
        color: item.riskLevel === 'critical' ? '#ef4444' : item.riskLevel === 'severe' ? '#f97316' : item.riskLevel === 'moderate' ? '#f59e0b' : '#10b981'
      }
    }));

    try {
      const isoSource = map.getSource('rain-isohyets-source') as mapboxgl.GeoJSONSource;
      if (isoSource) {
        isoSource.setData({
          type: 'FeatureCollection',
          features: isohyetFeatures,
        });
      }

      const anomalySource = map.getSource('rain-anomaly-source') as mapboxgl.GeoJSONSource;
      if (anomalySource) {
        anomalySource.setData({
          type: 'FeatureCollection',
          features: isohyetFeatures.filter(f => f.properties.efiPercentile > 60.0).map(f => ({
            ...f,
            properties: {
              ...f.properties,
              anomalyColor: '#c084fc'
            }
          })),
        });
      }

      const probSource = map.getSource('extreme-prob-source') as mapboxgl.GeoJSONSource;
      if (probSource) {
        probSource.setData({
          type: 'FeatureCollection',
          features: isohyetFeatures.map(f => ({
            type: 'Feature' as const,
            geometry: f.geometry,
            properties: {
              prob: f.properties.probGt50,
              color: f.properties.probGt50 > 80 ? '#ef4444' : f.properties.probGt50 > 40 ? '#f59e0b' : '#10b981'
            }
          })),
        });
      }
    } catch (e) {
      console.warn('GeoJSON live update skipped:', e);
    }
  }, [liveFeatures]);

  useEffect(() => {
    let isMounted = true;
    Promise.all([
      fetchApiThreatObjects(),
      fetchApiRiskGrid(selectedRegion)
    ]).then(([threats, grid]) => {
      if (isMounted) {
        setThreatObjects(threats);
        setRiskGrid(grid);
      }
    });
    return () => { isMounted = false; };
  }, [selectedRegion]);


  const toggleMapStyle = useCallback(() => {
    const nextStyle = mapStyle === 'dark' ? 'satellite' : 'dark';
    setMapStyle(nextStyle);
    if (mapRef.current) {
      mapRef.current.setStyle(getMapStyleSpec(nextStyle));
    }
  }, [mapStyle]);

  const timeSteps = useMemo(() => [
    { hour: -24, label: '-24h' },
    { hour: -12, label: '-12h' },
    { hour: 0, label: 'Now' },
    { hour: 3, label: '+3h' },
    { hour: 6, label: '+6h' },
    { hour: 12, label: '+12h' },
    { hour: 24, label: '+24h' },
    { hour: 48, label: '+48h' },
    { hour: 72, label: '+72h' },
  ], []);

  const clearMarkers = useCallback(() => {
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];
    popupsRef.current.forEach(p => p.remove());
    popupsRef.current = [];
  }, []);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    // Graceful Mapbox GL Access Token setup
    const token = API_CONFIG.mapboxPublicToken;
    if (token && token.startsWith('pk.')) {
      mapboxgl.accessToken = token;
    }

    // Disable Mapbox GL telemetry network events if no valid token
    try {
      if ((mapboxgl as any).config) {
        (mapboxgl as any).config.EVENTS_URL = null;
      }
    } catch (e) {}

    const map = new mapboxgl.Map({
      container: mapContainerRef.current!,
      style: getMapStyleSpec(mapStyle),
      center: [78.9629, 22.5937], // Center on India
      zoom: 4.8,
      pitch: is3DEnabled ? 40 : 0,
      bearing: is3DEnabled ? -10 : 0,
      projection: (API_CONFIG.mapboxPublicToken ? 'globe' : 'mercator') as any,
      antialias: true,
      maxZoom: 18,
      minZoom: 3,
    });

    map.addControl(new mapboxgl.NavigationControl({
      showCompass: true,
      showZoom: true,
      visualizePitch: true,
    }), 'bottom-right');

    map.addControl(new mapboxgl.ScaleControl({
      maxWidth: 120,
      unit: 'metric',
    }), 'bottom-left');

    map.on('error', (e: any) => {
      const msg = e?.error?.message || e?.message || String(e || '');
      const status = e?.error?.status;
      if (status === 401 || status === 403 || msg.includes('access token') || msg.includes('Unauthorized') || msg.includes('invalid Mapbox')) {
        console.warn('Mapbox token error encountered, automatically switching to free CartoDB dark basemap style.');
        try {
          map.setStyle(FREE_DARK_STYLE);
        } catch (err) {
          console.error('Failed to set fallback style:', err);
        }
      }
    });

    map.on('load', () => {
      map.resize();
      setTimeout(() => mapRef.current && mapRef.current.resize(), 200);
      setTimeout(() => mapRef.current && mapRef.current.resize(), 800);

      try {
        map.setFog({
          color: 'rgb(8, 12, 24)',
          'high-color': 'rgb(20, 30, 60)',
          'horizon-blend': 0.08,
          'space-color': 'rgb(4, 6, 12)',
          'star-intensity': 0.6,
        });
      } catch (e) {
        console.warn('Globe fog setting skipped:', e);
      }

      // --- 1. RainViewer & OpenWeatherMap Live Precipitation Radar Raster Layer ---
      map.addSource('rain-radar-source', {
        type: 'raster',
        tiles: [
          getOpenWeatherTileUrl('precipitation_new')
        ],
        tileSize: 256
      });

      map.addLayer({
        id: 'rain-radar-layer',
        type: 'raster',
        source: 'rain-radar-source',
        paint: {
          'raster-opacity': layerOpacity * 0.75,
          'raster-fade-duration': 300,
        },
      });

      // --- 2. Pan-India Downscaled Rainfall Isohyet GeoJSON Layer ---
      const isohyetFeatures = liveFeatures.map((item, idx) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Polygon' as const,
          coordinates: [item.coords],
        },
        properties: {
          id: `ISO-${idx + 1}`,
          name: item.name,
          district: item.district,
          state: item.state,
          rainMm: item.rainMm,
          efiPercentile: item.efiPercentile,
          probGt50: item.probGt50,
          riskLevel: item.riskLevel,
          color: item.riskLevel === 'critical' ? '#ef4444' : item.riskLevel === 'severe' ? '#f97316' : item.riskLevel === 'moderate' ? '#f59e0b' : '#10b981'
        }
      }));

      map.addSource('rain-isohyets-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: isohyetFeatures,
        },
      });

      map.addLayer({
        id: 'rain-isohyets-fill',
        type: 'fill',
        source: 'rain-isohyets-source',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': layerOpacity * 0.45,
        },
      });

      map.addLayer({
        id: 'rain-isohyets-line',
        type: 'line',
        source: 'rain-isohyets-source',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 2.0,
          'line-opacity': 0.85,
        },
      });

      // --- 3. Extreme Forecast Index (EFI) Anomaly Contours Layer ---
      map.addSource('rain-anomaly-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: isohyetFeatures.filter(f => f.properties.efiPercentile > 96.0).map(f => ({
            ...f,
            properties: {
              ...f.properties,
              anomalyColor: '#c084fc'
            }
          }))
        }
      });

      map.addLayer({
        id: 'rain-anomaly-fill',
        type: 'fill',
        source: 'rain-anomaly-source',
        paint: {
          'fill-color': '#a855f7',
          'fill-opacity': layerOpacity * 0.35,
        }
      });

      map.addLayer({
        id: 'rain-anomaly-line',
        type: 'line',
        source: 'rain-anomaly-source',
        paint: {
          'line-color': '#e879f9',
          'line-width': 2.5,
          'line-dasharray': [2, 2],
          'line-opacity': 0.9,
        }
      });

      // --- 4. Exceedance Probability Layer (>50mm/24h) ---
      map.addSource('extreme-prob-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: isohyetFeatures.map(f => ({
            type: 'Feature' as const,
            geometry: f.geometry,
            properties: {
              prob: f.properties.probGt50,
              color: f.properties.probGt50 > 90 ? '#ef4444' : '#f59e0b'
            }
          }))
        }
      });

      map.addLayer({
        id: 'extreme-prob-fill',
        type: 'fill',
        source: 'extreme-prob-source',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': layerOpacity * 0.2,
        }
      });

      // --- 5. 5km Grid Cells Overlay ---
      const gridFeatures = riskGrid.map((cell: GridCell5km) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Polygon' as const,
          coordinates: [[
            [cell.lng - 0.025, cell.lat - 0.025],
            [cell.lng + 0.025, cell.lat - 0.025],
            [cell.lng + 0.025, cell.lat + 0.025],
            [cell.lng - 0.025, cell.lat + 0.025],
            [cell.lng - 0.025, cell.lat - 0.025],
          ]],
        },
        properties: {
          id: cell.id,
          riskLevel: cell.riskLevel,
          riskScore: cell.downscaledRiskScore,
          rainfall: cell.rainfallForecastMm,
          probability: cell.probabilityGt50mm,
          district: cell.district,
          tehsil: cell.tehsil,
          elevation: cell.elevationMeters,
          vulnerability: cell.vulnerabilityIndex,
          anomaly: cell.anomalyPercentile,
          color: RISK_COLORS[cell.riskLevel] || '#10b981',
          height: cell.downscaledRiskScore * 40,
        },
      }));

      map.addSource('risk-grid', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: gridFeatures,
        },
      });

      map.addLayer({
        id: 'risk-grid-3d',
        type: 'fill-extrusion',
        source: 'risk-grid',
        paint: {
          'fill-extrusion-color': ['get', 'color'],
          'fill-extrusion-height': is3DEnabled ? ['get', 'height'] : 0,
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': layerOpacity * 0.5,
        },
      });

      map.addLayer({
        id: 'risk-grid-flat',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': layerOpacity * 0.25,
        },
      });

      map.addLayer({
        id: 'risk-grid-outline',
        type: 'line',
        source: 'risk-grid',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 0.8,
          'line-opacity': 0.4,
        },
      });

      // --- 6. Threat Footprints GeoJSON ---
      const threatFeatures = threatObjects.map((threat: ThreatObject) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Polygon' as const,
          coordinates: [threat.polygonCoords.map(([lat, lng]: [number, number]) => [lng, lat]).concat([
            [threat.polygonCoords[0][1], threat.polygonCoords[0][0]]
          ])],
        },
        properties: {
          id: threat.id,
          name: threat.name,
          riskLevel: threat.riskLevel,
          color: RISK_COLORS[threat.riskLevel] || '#f59e0b',
          district: threat.district,
          peakIntensity: threat.peakIntensityMmH,
          probability: threat.probabilityExceedance,
          speed: threat.speedKmH,
          direction: threat.direction,
          height: threat.probabilityExceedance * 70,
        },
      }));

      map.addSource('threat-footprints', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: threatFeatures,
        },
      });

      map.addLayer({
        id: 'threat-fill',
        type: 'fill',
        source: 'threat-footprints',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': layerOpacity * 0.35,
        },
      });

      map.addLayer({
        id: 'threat-outline',
        type: 'line',
        source: 'threat-footprints',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 2.5,
          'line-dasharray': [3, 3],
          'line-opacity': 0.85,
        },
      });

      map.addLayer({
        id: 'threat-3d',
        type: 'fill-extrusion',
        source: 'threat-footprints',
        paint: {
          'fill-extrusion-color': ['get', 'color'],
          'fill-extrusion-height': is3DEnabled ? ['get', 'height'] : 0,
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': layerOpacity * 0.4,
        },
      });

      // --- 7. Trajectories & Markers ---
      const trajectoryFeatures = threatObjects.map((threat: ThreatObject) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'LineString' as const,
          coordinates: threat.trajectoryPoints.map((p: any) => [p.lng, p.lat]),
        },
        properties: {
          id: threat.id,
          name: threat.name,
        },
      }));

      map.addSource('trajectories', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: trajectoryFeatures,
        },
      });

      map.addLayer({
        id: 'trajectory-line',
        type: 'line',
        source: 'trajectories',
        paint: {
          'line-color': '#06b6d4',
          'line-width': 3,
          'line-dasharray': [4, 4],
          'line-opacity': 0.85,
        },
      });

      const waypointFeatures = threatObjects.flatMap((threat: ThreatObject) => 
        threat.trajectoryPoints.map((p: any, idx: number) => ({
          type: 'Feature' as const,
          geometry: {
            type: 'Point' as const,
            coordinates: [p.lng, p.lat],
          },
          properties: {
            isOrigin: idx === 0,
            timestamp: p.timestamp,
            hour: p.forecastHour,
            riskLevel: p.riskLevel,
            color: idx === 0 ? '#ef4444' : RISK_COLORS[p.riskLevel] || '#06b6d4',
          },
        }))
      );

      map.addSource('waypoints', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: waypointFeatures,
        },
      });

      map.addLayer({
        id: 'waypoint-circles',
        type: 'circle',
        source: 'waypoints',
        paint: {
          'circle-radius': ['case', ['get', 'isOrigin'], 8, 5],
          'circle-color': ['get', 'color'],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
          'circle-opacity': 0.95,
        },
      });

      // --- 8. High Speed Wind & Cyclones Overlay Layer ---
      const windExtremesFeatures = [
        {
          type: 'Feature' as const,
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[
              [82.0, 12.0], [88.0, 12.0], [89.0, 15.0], [88.0, 18.0], [82.0, 18.0], [81.0, 15.0], [82.0, 12.0]
            ]]
          },
          properties: {
            id: 'WIND-CYC-01',
            name: 'Super Cyclone Amphan 220 km/h Wind Radius',
            windSpeed: 220,
            type: 'cyclone',
            color: '#ef4444'
          }
        },
        {
          type: 'Feature' as const,
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[
              [83.0, 24.5], [86.0, 24.5], [86.5, 26.5], [83.5, 26.5], [83.0, 24.5]
            ]]
          },
          properties: {
            id: 'WIND-SQUALL-02',
            name: 'Pre-Monsoon Squall Line 135 km/h Gust Band',
            windSpeed: 135,
            type: 'squall',
            color: '#f59e0b'
          }
        },
        {
          type: 'Feature' as const,
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[
              [72.5, 18.5], [73.5, 18.5], [73.8, 19.8], [72.8, 19.8], [72.5, 18.5]
            ]]
          },
          properties: {
            id: 'WIND-GALE-03',
            name: 'Konkan Coast Gale Force Wind Zone',
            windSpeed: 110,
            type: 'gale',
            color: '#f97316'
          }
        }
      ];

      map.addSource('wind-extremes-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: windExtremesFeatures
        }
      });

      map.addLayer({
        id: 'wind-extremes-fill',
        type: 'fill',
        source: 'wind-extremes-source',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': layerOpacity * 0.35
        }
      });

      map.addLayer({
        id: 'wind-extremes-line',
        type: 'line',
        source: 'wind-extremes-source',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 3,
          'line-dasharray': [2, 2],
          'line-opacity': 0.9
        }
      });

      // --- 9. State / District Boundaries Layer ---
      const adminBoundaryFeatures = [
        {
          name: 'Uttar Pradesh State & District Boundaries',
          district: 'Prayagraj / Varanasi / Lucknow',
          coords: [[80.5, 24.5], [83.5, 24.5], [84.0, 27.5], [80.0, 27.5], [80.5, 24.5]]
        },
        {
          name: 'Maharashtra State & Coastal Boundaries',
          district: 'Mumbai Suburban / Thane / Raigad',
          coords: [[72.5, 18.5], [74.5, 18.5], [74.5, 20.5], [72.5, 20.5], [72.5, 18.5]]
        },
        {
          name: 'Kerala Malabar & High Range Boundaries',
          district: 'Wayanad / Kozhikode / Malappuram',
          coords: [[75.5, 11.0], [77.2, 11.0], [77.2, 12.5], [75.5, 12.5], [75.5, 11.0]]
        },
        {
          name: 'Assam Valley & Northeast Boundaries',
          district: 'Kamrup / Guwahati / Dibrugarh',
          coords: [[89.8, 25.0], [95.5, 25.0], [95.5, 28.0], [89.8, 28.0], [89.8, 25.0]]
        },
        {
          name: 'Odisha Coastal Belt Boundaries',
          district: 'Cuttack / Puri / Balasore',
          coords: [[84.5, 19.5], [87.5, 19.5], [87.5, 22.0], [84.5, 22.0], [84.5, 19.5]]
        }
      ].map((item, idx) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Polygon' as const,
          coordinates: [item.coords]
        },
        properties: {
          id: `ADMIN-${idx + 1}`,
          name: item.name,
          district: item.district,
          color: '#38bdf8'
        }
      }));

      map.addSource('admin-boundaries-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: adminBoundaryFeatures
        }
      });

      map.addLayer({
        id: 'admin-districts-line',
        type: 'line',
        source: 'admin-boundaries-source',
        paint: {
          'line-color': '#38bdf8',
          'line-width': 1.8,
          'line-dasharray': [3, 1],
          'line-opacity': 0.85
        }
      });

      map.addLayer({
        id: 'admin-states-line',
        type: 'line',
        source: 'admin-boundaries-source',
        paint: {
          'line-color': '#60a5fa',
          'line-width': 2.5,
          'line-opacity': 0.95
        }
      });

      // --- 10. River Basins & Orographic Slope Vulnerability Layer ---
      const riverBasinFeatures = [
        {
          name: 'Kosi River Catchment & Floodplain Inundation Zone',
          basin: 'Kosi Basin (Bihar)',
          vulnerability: 'Critical Slope Siltation Risk',
          coords: [[86.2, 25.5], [87.5, 25.5], [87.5, 26.8], [86.2, 26.8], [86.2, 25.5]]
        },
        {
          name: 'Yamuna-Ganges Sangam River Basin Zone',
          basin: 'Ganga-Yamuna Basin (Prayagraj)',
          vulnerability: 'High Urban Confluence Surge Risk',
          coords: [[81.2, 25.0], [82.5, 25.0], [82.5, 25.8], [81.2, 25.8], [81.2, 25.0]]
        },
        {
          name: 'Western Ghats Wayanad Slope Landslide Corridor',
          basin: 'Periyar-Kabini Catchment',
          vulnerability: 'Extreme Orographic Slope Debris Flow',
          coords: [[75.8, 11.3], [76.6, 11.3], [76.6, 12.1], [75.8, 12.1], [75.8, 11.3]]
        },
        {
          name: 'Brahmaputra Middle Valley Inundation Basin',
          basin: 'Brahmaputra Basin (Guwahati)',
          vulnerability: 'Severe Riverine Embankment Breach Risk',
          coords: [[91.0, 25.8], [92.8, 25.8], [92.8, 26.8], [91.0, 26.8], [91.0, 25.8]]
        }
      ].map((item, idx) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Polygon' as const,
          coordinates: [item.coords]
        },
        properties: {
          id: `BASIN-${idx + 1}`,
          name: item.name,
          basin: item.basin,
          vulnerability: item.vulnerability,
          color: '#06b6d4'
        }
      }));

      map.addSource('vulnerability-basins-source', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: riverBasinFeatures
        }
      });

      map.addLayer({
        id: 'river-basins-fill',
        type: 'fill',
        source: 'vulnerability-basins-source',
        paint: {
          'fill-color': '#0284c7',
          'fill-opacity': layerOpacity * 0.25
        }
      });

      map.addLayer({
        id: 'river-basins-line',
        type: 'line',
        source: 'vulnerability-basins-source',
        paint: {
          'line-color': '#38bdf8',
          'line-width': 2.0,
          'line-opacity': 0.8
        }
      });

      map.addLayer({
        id: 'slope-zones-line',
        type: 'line',
        source: 'vulnerability-basins-source',
        paint: {
          'line-color': '#06b6d4',
          'line-width': 2.5,
          'line-dasharray': [2, 2],
          'line-opacity': 0.9
        }
      });

      // Add HTML markers for threat centroids with dynamic hazard badges (Cyclone, Wind Squall, Heat Dome, Rain Cell)
      threatObjects.forEach((threat: ThreatObject) => {
        const el = document.createElement('div');
        el.className = 'threat-marker-container';

        const isCyclone = threat.id.includes('CYC') || (threat.hazardType && threat.hazardType.toLowerCase().includes('cyclone')) || threat.name.toLowerCase().includes('cyclone');
        const isWind = threat.id.includes('WIND') || (threat.hazardType && threat.hazardType.toLowerCase().includes('wind')) || threat.name.toLowerCase().includes('squall');
        const isHeat = threat.id.includes('HEAT') || (threat.hazardType && threat.hazardType.toLowerCase().includes('heat'));

        let badgeBg = 'rgba(15, 23, 42, 0.92)';
        let borderCol = RISK_COLORS[threat.riskLevel] || '#ef4444';
        let glowCol = RISK_GLOW[threat.riskLevel] || 'rgba(239, 68, 68, 0.4)';
        let metricText = `${threat.peakIntensityMmH} mm/h Rain`;
        let iconHtml = `<span style="font-size: 16px;">🌧️</span>`;

        if (isCyclone) {
          badgeBg = 'rgba(127, 29, 29, 0.92)';
          borderCol = '#ef4444';
          glowCol = 'rgba(239, 68, 68, 0.7)';
          metricText = threat.hazardMetricDisplay || `${threat.speedKmH} km/h Cat 5`;
          iconHtml = `<span style="display:inline-block; animation: spin 3s linear infinite; font-size: 18px;">🌀</span>`;
        } else if (isWind) {
          badgeBg = 'rgba(120, 53, 15, 0.92)';
          borderCol = '#f59e0b';
          glowCol = 'rgba(245, 158, 11, 0.7)';
          metricText = threat.hazardMetricDisplay || `${threat.speedKmH} km/h Gusts`;
          iconHtml = `<span style="font-size: 18px;">💨</span>`;
        } else if (isHeat) {
          badgeBg = 'rgba(124, 45, 18, 0.92)';
          borderCol = '#ea580c';
          glowCol = 'rgba(234, 88, 12, 0.7)';
          metricText = threat.hazardMetricDisplay || '50°C Anomaly';
          iconHtml = `<span style="font-size: 18px;">🌡️</span>`;
        }

        el.innerHTML = `
          <div style="
            display: flex;
            align-items: center;
            gap: 6px;
            background: ${badgeBg};
            border: 2px solid ${borderCol};
            border-radius: 20px;
            padding: 4px 10px;
            box-shadow: 0 0 20px ${glowCol}, 0 4px 12px rgba(0,0,0,0.5);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            white-space: nowrap;
            backdrop-filter: blur(8px);
          ">
            ${iconHtml}
            <div style="display: flex; flex-direction: column; text-align: left; line-height: 1.15;">
              <span style="font-size: 11px; font-weight: 800; color: #f8fafc; font-family: system-ui, sans-serif; letter-spacing: 0.01em;">
                ${threat.name}
              </span>
              <span style="font-size: 10px; font-weight: 700; color: #38bdf8; font-family: monospace;">
                ${metricText} • ${threat.district}
              </span>
            </div>
          </div>
        `;
        el.style.cursor = 'pointer';

        const popup = new mapboxgl.Popup({
          offset: 20,
          closeButton: true,
          maxWidth: '320px',
        }).setHTML(`
          <div style="font-family: 'Inter', system-ui, sans-serif; padding: 6px;">
            <div style="font-weight: 800; font-size: 13px; color: ${borderCol}; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
              <span>${isCyclone ? '🌀' : isWind ? '💨' : isHeat ? '🌡️' : '🌧️'}</span>
              <span>${threat.name}</span>
            </div>
            <div style="font-size: 11px; color: #b4c1db; line-height: 1.6;">
              <strong style="color: #f0f4ff;">District:</strong> ${threat.district}<br/>
              <strong style="color: #f0f4ff;">Region:</strong> ${threat.region}<br/>
              <strong style="color: #f0f4ff;">Hazard Type:</strong> <span style="color: #38bdf8; font-weight:700;">${threat.hazardType || 'Convective Precipitation Cell'}</span><br/>
              <strong style="color: #f0f4ff;">Intensity / Metric:</strong> <span style="color: #f59e0b; font-weight:700;">${metricText}</span><br/>
              <strong style="color: #f0f4ff;">Track Speed:</strong> ${threat.speedKmH} km/h (${threat.direction})<br/>
              <strong style="color: #f0f4ff;">5km Subgrid Quantile:</strong> 
                <span style="color: ${borderCol}; font-weight: 700;">${threat.probabilityExceedance}% Exceedance</span>
            </div>
          </div>
        `);

        popupsRef.current.push(popup);

        const marker = new mapboxgl.Marker({
          element: el,
          anchor: 'center',
        })
          .setLngLat([threat.centroid[1], threat.centroid[0]])
          .setPopup(popup)
          .addTo(map);

        el.addEventListener('click', () => {
          if (onSelectThreat) onSelectThreat(threat);
        });

        markersRef.current.push(marker);
      });

      // Hover / Click interaction for Rain Isohyet Polygons
      map.on('click', 'rain-isohyets-fill', (e) => {
        if (!e.features?.length) return;
        const props = (e.features[0] as any).properties;
        if (!props) return;

        const rainVal = Number(props.rainMm || 0);
        const rainColor = rainVal >= 50 ? '#ef4444' : rainVal >= 15 ? '#f97316' : rainVal > 0 ? '#f59e0b' : '#10b981';

        new mapboxgl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="font-family: sans-serif; font-size: 12px; padding: 6px;">
              <div style="font-weight: bold; color: #38bdf8; font-size: 13px; margin-bottom: 4px;">🌧️ ${props.name}</div>
              <div><strong>District:</strong> ${props.district} (${props.state})</div>
              <div><strong>Live Rainfall (Open-Meteo):</strong> <span style="color: ${rainColor}; font-weight: bold;">${rainVal} mm</span></div>
              <div><strong>EFI Climatological Percentile:</strong> ${props.efiPercentile}th</div>
              <div><strong>5km Quantile Exceedance Prob:</strong> ${props.probGt50}%</div>
              <div style="margin-top: 6px; font-size: 10px; color: #10b981; font-weight: bold; border-top: 1px solid #1e2d48; padding-top: 4px;">
                🟢 Live ECMWF / Open-Meteo Telemetry
              </div>
            </div>
          `)
          .addTo(map);
      });

      // Click interaction for River Basins
      map.on('click', 'river-basins-fill', (e) => {
        if (!e.features?.length) return;
        const props = (e.features[0] as any).properties;
        if (!props) return;

        new mapboxgl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="font-family: sans-serif; font-size: 12px; padding: 4px;">
              <div style="font-weight: bold; color: #06b6d4; font-size: 13px; margin-bottom: 4px;">🌊 ${props.name}</div>
              <div><strong>Catchment Basin:</strong> ${props.basin}</div>
              <div><strong>Slope Vulnerability:</strong> <span style="color: #38bdf8; font-weight: bold;">${props.vulnerability}</span></div>
            </div>
          `)
          .addTo(map);
      });

      map.on('mouseenter', 'rain-isohyets-fill', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'rain-isohyets-fill', () => {
        map.getCanvas().style.cursor = '';
      });
      map.on('mouseenter', 'river-basins-fill', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'river-basins-fill', () => {
        map.getCanvas().style.cursor = '';
      });

      map.on('click', 'risk-grid-flat', (e) => {
        if (!e.features?.length) return;
        const props = (e.features[0] as any).properties;
        if (!props) return;
        const cell = riskGrid.find((c: GridCell5km) => c.id === props.id);
        if (cell) {
          setSelectedCell(cell);
          setShowCellPanel(true);
        }
      });
    });

    mapRef.current = map;

    return () => {
      clearMarkers();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update region fly-to
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const preset = INDIA_REGION_PRESETS.find(p => p.id === currentRegion) || INDIA_REGION_PRESETS[0];
    map.flyTo({
      center: [preset.center[1], preset.center[0]],
      zoom: preset.zoom,
      pitch: is3DEnabled ? 40 : 0,
      bearing: is3DEnabled ? -10 : 0,
      duration: 1800,
      essential: true,
    });
  }, [currentRegion, is3DEnabled]);

  // Sync layer visibilities dynamically for ALL 8 GIS LAYERS
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const layerMap: Record<string, string[]> = {
      rainfall_forecast: ['rain-radar-layer', 'rain-isohyets-fill', 'rain-isohyets-line'],
      rainfall_anomaly: ['rain-anomaly-fill', 'rain-anomaly-line'],
      extreme_probability: ['extreme-prob-fill'],
      risk_grid_5km: ['risk-grid-3d', 'risk-grid-flat', 'risk-grid-outline'],
      threat_footprint: ['threat-fill', 'threat-outline', 'threat-3d'],
      trajectory: ['trajectory-line', 'waypoint-circles'],
      admin_boundaries: ['admin-districts-line', 'admin-states-line'],
      vulnerability: ['river-basins-fill', 'river-basins-line', 'slope-zones-line'],
    };

    Object.entries(layerMap).forEach(([key, mapboxLayers]) => {
      const visible = activeLayers[key as MapLayerId];
      mapboxLayers.forEach(layerId => {
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
        }
      });
    });
  }, [activeLayers]);

  // Sync opacity
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const opacityUpdates: [string, string, number][] = [
      ['rain-radar-layer', 'raster-opacity', layerOpacity * 0.75],
      ['rain-isohyets-fill', 'fill-opacity', layerOpacity * 0.45],
      ['rain-anomaly-fill', 'fill-opacity', layerOpacity * 0.35],
      ['risk-grid-3d', 'fill-extrusion-opacity', layerOpacity * 0.5],
      ['risk-grid-flat', 'fill-opacity', layerOpacity * 0.25],
      ['threat-fill', 'fill-opacity', layerOpacity * 0.35],
      ['threat-3d', 'fill-extrusion-opacity', layerOpacity * 0.4],
      ['wind-extremes-fill', 'fill-opacity', layerOpacity * 0.35],
    ];

    opacityUpdates.forEach(([layerId, prop, value]) => {
      if (map.getLayer(layerId)) {
        map.setPaintProperty(layerId, prop as any, value);
      }
    });
  }, [layerOpacity]);

  const toggle3D = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const newState = !is3DEnabled;
    setIs3DEnabled(newState);

    if (newState) {
      map.easeTo({ pitch: 40, bearing: -10, duration: 1200 });
    } else {
      map.easeTo({ pitch: 0, bearing: 0, duration: 1200 });
    }
  }, [is3DEnabled]);

  const resetView = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    map.flyTo({
      center: [78.9629, 22.5937],
      zoom: 4.8,
      pitch: is3DEnabled ? 40 : 0,
      bearing: is3DEnabled ? -10 : 0,
      duration: 1800,
    });
    setCurrentRegion('all');
  }, [is3DEnabled]);

  useEffect(() => {
    const el = mapContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      if (mapRef.current) {
        mapRef.current.resize();
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const toggleLayer = useCallback((layerId: MapLayerId) => {
    setActiveLayers(prev => ({ ...prev, [layerId]: !prev[layerId] }));
  }, []);

  const layerLabels: Partial<Record<MapLayerId, { label: string; icon: string }>> = {
    rainfall_forecast: { label: 'Live Pan-India Rainfall Radar', icon: '🌧️' },
    rainfall_anomaly: { label: 'EFI Climatology Anomaly', icon: '⚡' },
    extreme_probability: { label: 'Extreme Prob (>50mm)', icon: '🎯' },
    threat_footprint: { label: 'Threat Polygons', icon: '🛡️' },
    trajectory: { label: 'GNN Trajectory Track', icon: '↗️' },
    risk_grid_5km: { label: '5 km Risk Grid Overlay', icon: '📐' },
    admin_boundaries: { label: 'State/District Boundaries', icon: '🏛️' },
    vulnerability: { label: 'River Basins & Slope Zones', icon: '🌊' },
  };

  const activeCount = useMemo(() => {
    return Object.keys(layerLabels).filter(k => activeLayers[k as MapLayerId]).length;
  }, [activeLayers]);

  return (
    <div className={`relative flex flex-col h-full w-full min-h-[450px] sm:min-h-[500px] ${isFullscreen ? 'fixed inset-0 z-50 bg-[#070b16] p-0' : 'overflow-hidden border border-[#1a2540]'}`}>
      {/* Map Header Bar */}
      <div className="bg-[#0a0f1e]/95 backdrop-blur-xl border-b border-[#1a2540] px-2.5 sm:px-4 py-2 flex flex-wrap items-center justify-between gap-2 z-20">
        <div className="flex items-center gap-2 min-w-0">
          <div className="relative shrink-0">
            <div className="h-2.5 w-2.5 sm:h-3 sm:w-3 rounded-full bg-cyan-400"></div>
            <div className="absolute inset-0 h-2.5 w-2.5 sm:h-3 sm:w-3 rounded-full bg-cyan-400 animate-ping opacity-75"></div>
          </div>
          <div className="min-w-0">
            <h2 className="text-xs sm:text-sm font-bold text-slate-100 flex items-center gap-1.5 truncate">
              <span className="truncate">StormTrace 3D GIS Engine</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50 font-mono shrink-0">
                Live Doppler Radar
              </span>
              <span className="hidden md:inline text-[9px] px-1.5 py-0.5 rounded bg-violet-950/60 text-violet-300 border border-violet-800/40 font-mono shrink-0">
                DDPM 5km
              </span>
            </h2>
            <p className="hidden sm:block text-[10px] sm:text-[11px] text-slate-400 truncate">
              Spherical GNN Anomaly Tracker • 5km Diffusion Downscaled Extreme Value Radar
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-0.5 max-w-full">
          {/* Region Selector */}
          <div className="flex items-center gap-1 bg-[#111827] border border-[#1e2d48] px-2 py-1 rounded-lg text-xs text-slate-200 shrink-0">
            <Globe className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
            <select
              value={currentRegion}
              onChange={(e) => setCurrentRegion(e.target.value as IndiaRegionId)}
              className="bg-transparent text-[11px] text-slate-200 font-bold focus:outline-none cursor-pointer max-w-[80px] sm:max-w-none"
            >
              {INDIA_REGION_PRESETS.map(p => (
                <option key={p.id} value={p.id} className="bg-[#111827]">{p.name}</option>
              ))}
            </select>
          </div>

          {/* Style Toggle */}
          <button
            onClick={toggleMapStyle}
            className="px-2 py-1 rounded-lg bg-[#111827] hover:bg-[#1e2d48] border border-[#1e2d48] text-[11px] font-semibold text-slate-300 flex items-center gap-1 transition-all shrink-0"
            title="Switch Map Base Style"
          >
            <Globe className="h-3 w-3 text-cyan-400 shrink-0" />
            <span className="capitalize font-mono text-[10px]">{mapStyle}</span>
          </button>

          {/* 3D Toggle */}
          <button
            onClick={toggle3D}
            className={`p-1 sm:p-1.5 rounded-lg border transition-all shrink-0 ${
              is3DEnabled 
                ? 'bg-cyan-600/20 border-cyan-500/40 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.2)]' 
                : 'bg-[#111827] border-[#1e2d48] text-slate-400 hover:text-slate-200'
            }`}
            title={is3DEnabled ? 'Disable 3D View' : 'Enable 3D View'}
          >
            <Mountain className="h-3.5 w-3.5" />
          </button>

          {/* Reset View */}
          <button
            onClick={resetView}
            className="p-1 sm:p-1.5 rounded-lg bg-[#111827] hover:bg-[#1e2d48] border border-[#1e2d48] text-slate-400 hover:text-slate-200 transition-all shrink-0"
            title="Reset to Pan-India View"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>

          {/* Opacity Slider */}
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 bg-[#111827] border border-[#1e2d48] px-2 py-1 rounded-lg shrink-0">
            <Sliders className="h-3.5 w-3.5 text-cyan-400" />
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.05"
              value={layerOpacity}
              onChange={(e) => setLayerOpacity(parseFloat(e.target.value))}
              className="w-16 accent-cyan-400 cursor-pointer"
            />
            <span className="font-mono text-cyan-300 w-6 text-[11px]">{Math.round(layerOpacity * 100)}%</span>
          </div>

          {/* Fullscreen */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 sm:p-1.5 rounded-lg bg-[#111827] hover:bg-[#1e2d48] border border-[#1e2d48] text-slate-300 transition-all shrink-0"
            title="Toggle Fullscreen"
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Map Canvas */}
      <div className="relative flex-1 w-full h-full min-h-[350px] sm:min-h-[500px] bg-[#060a14] overflow-hidden">
        <div ref={mapContainerRef} className="absolute inset-0 z-0 w-full h-full" />

        {/* Left Layer Panel Toggle (Left Eye Button) */}
        <button
          onClick={() => {
            setShowLayerPanel(!showLayerPanel);
            if (!showLayerPanel && window.innerWidth < 640) {
              setShowCellPanel(false); // Auto close cell panel on phone screens to prevent overlapping
            }
          }}
          className="absolute top-2 left-2 sm:top-4 sm:left-4 z-10 p-1.5 rounded-lg bg-[#0a0f1e]/90 backdrop-blur-lg border border-[#1e2d48] text-slate-300 hover:text-white transition-all shadow-lg flex items-center gap-1.5 text-xs font-semibold"
          title={showLayerPanel ? 'Hide Left Layer Panel' : 'Show Left Layer Panel'}
        >
          {showLayerPanel ? <EyeOff className="h-3.5 w-3.5 text-cyan-400" /> : <Eye className="h-3.5 w-3.5 text-cyan-400" />}
          <span className="text-[10px] text-slate-300">Layers</span>
        </button>

        {/* Right Cell Panel Toggle (Right Eye Button) */}
        {selectedCell && (
          <button
            onClick={() => {
              setShowCellPanel(!showCellPanel);
              if (!showCellPanel && window.innerWidth < 640) {
                setShowLayerPanel(false); // Auto close layer panel on phone screens to prevent overlapping
              }
            }}
            className="absolute top-2 right-2 sm:top-4 sm:right-4 z-10 p-1.5 rounded-lg bg-[#0a0f1e]/90 backdrop-blur-lg border border-cyan-500/40 text-cyan-300 hover:text-white transition-all shadow-lg flex items-center gap-1.5 text-xs font-semibold"
            title={showCellPanel ? 'Hide Right Cell Panel' : 'Show Right Cell Panel'}
          >
            {showCellPanel ? <EyeOff className="h-3.5 w-3.5 text-cyan-400" /> : <Eye className="h-3.5 w-3.5 text-cyan-400" />}
            <span className="text-[10px] text-cyan-300">Cell Details</span>
          </button>
        )}

        {/* Left Floating Layer Controls Panel */}
        {showLayerPanel && (
          <div className="absolute top-10 left-2 sm:top-12 sm:left-4 z-10 w-[calc(100vw-2rem)] max-w-[240px] sm:w-64 bg-[#0a0f1e]/95 backdrop-blur-xl p-2 sm:p-2.5 rounded-xl border border-[#1e2d48] max-h-[50vh] sm:max-h-[60vh] overflow-y-auto shadow-2xl space-y-1.5 text-[11px]">
            <div className="flex items-center justify-between pb-1.5 border-b border-[#1e2d48]">
              <span className="text-[11px] font-bold text-slate-200 flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-cyan-400" />
                Active GIS Layers
              </span>
              <span className="text-[9px] text-cyan-400 font-mono">
                {activeCount}/8 Active
              </span>
            </div>

            <div className="space-y-0.5">
              {(Object.keys(layerLabels) as MapLayerId[]).map(layerId => {
                const info = layerLabels[layerId];
                if (!info) return null;
                const isChecked = !!activeLayers[layerId];
                return (
                  <button
                    key={layerId}
                    onClick={() => toggleLayer(layerId)}
                    className={`w-full text-left px-2 py-1.5 rounded-md text-[11px] flex items-center justify-between transition-all ${
                      isChecked
                        ? 'bg-cyan-950/60 border border-cyan-500/30 text-cyan-200 font-medium'
                        : 'bg-[#111827]/60 border border-[#1e2d48]/80 text-slate-400 hover:text-slate-200 hover:bg-[#151d33]'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 truncate">
                      <span className="text-xs">{info.icon}</span>
                      <span className="truncate">{info.label}</span>
                    </div>
                    {isChecked ? (
                      <Eye className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                    ) : (
                      <EyeOff className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>

            <div className="mt-2 pt-1.5 border-t border-[#1e2d48] text-[10px]">
              <span className="text-slate-400 block font-semibold mb-1">Precipitation Legend:</span>
              <div className="grid grid-cols-4 gap-1 text-[9px] text-center font-bold">
                <div className="py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/40">&lt;50mm</div>
                <div className="py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/40">50-100</div>
                <div className="py-0.5 rounded bg-orange-950/60 text-orange-400 border border-orange-800/40">100-150</div>
                <div className="py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/40">&gt;150mm</div>
              </div>
            </div>
          </div>
        )}

        {/* Right Selected Cell Inspection Panel */}
        {showCellPanel && selectedCell && (
          <div className="absolute top-10 right-2 sm:top-12 sm:right-4 z-10 w-[calc(100vw-2rem)] max-w-[240px] sm:w-64 bg-[#0a0f1e]/95 backdrop-blur-xl p-2 sm:p-2.5 rounded-xl border border-cyan-500/30 shadow-2xl space-y-1.5 text-[11px]">
            <div className="flex items-center justify-between pb-1 border-b border-[#1e2d48]">
              <div>
                <span className="text-[9px] font-mono text-cyan-400 block uppercase tracking-wider">5 KM SUB-GRID CELL</span>
                <h4 className="font-bold text-slate-100 text-xs truncate max-w-[150px]">{selectedCell.tehsil} ({selectedCell.district})</h4>
              </div>
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold uppercase ${
                selectedCell.riskLevel === 'critical' ? 'bg-red-600 text-white' :
                selectedCell.riskLevel === 'severe' ? 'bg-orange-600 text-white' : 'bg-amber-600 text-slate-900'
              }`}>
                {selectedCell.riskLevel}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-1.5 text-[10px]">
              <div className="bg-[#111827]/80 p-1.5 rounded-md border border-[#1e2d48]">
                <span className="text-slate-500 block text-[9px]">Coordinates</span>
                <span className="text-slate-200 font-mono text-[10px]">{selectedCell.lat}°N, {selectedCell.lng}°E</span>
              </div>
              <div className="bg-[#111827]/80 p-1.5 rounded-md border border-[#1e2d48]">
                <span className="text-slate-500 block text-[9px]">Rainfall (24h)</span>
                <span className="text-cyan-300 font-bold text-[11px]">{selectedCell.rainfallForecastMm} mm</span>
              </div>
              <div className="bg-[#111827]/80 p-1.5 rounded-md border border-[#1e2d48]">
                <span className="text-slate-500 block text-[9px]">EFI Percentile</span>
                <span className="text-amber-400 font-bold text-[11px]">{selectedCell.anomalyPercentile}th</span>
              </div>
              <div className="bg-[#111827]/80 p-1.5 rounded-md border border-[#1e2d48]">
                <span className="text-slate-500 block text-[9px]">Prob &gt; 50mm</span>
                <span className="text-red-400 font-bold text-[11px]">{selectedCell.probabilityGt50mm}%</span>
              </div>
            </div>

            <div className="text-[10px] text-slate-300 bg-[#111827]/60 p-1.5 rounded-md border border-[#1e2d48] flex justify-between">
              <span>DEM Elev: <strong>{selectedCell.elevationMeters}m</strong></span>
              <span>Flood Idx: <strong className="text-cyan-400">{selectedCell.vulnerabilityIndex}</strong></span>
            </div>
          </div>
        )}

        {is3DEnabled && (
          <div className="hidden sm:flex absolute bottom-4 left-4 z-10 items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0a0f1e]/85 backdrop-blur-lg border border-violet-500/30 text-[11px] text-violet-300 font-mono">
            <Compass className="h-3.5 w-3.5 text-violet-400 animate-spin" style={{ animationDuration: '8s' }} />
            3D Globe View Active • GIS Radar Overlay
          </div>
        )}
      </div>

      {/* Bottom Time Step Bar */}
      <div className="bg-[#0a0f1e]/95 backdrop-blur-xl border-t border-[#1a2540] p-2 sm:p-3 flex flex-col sm:flex-row items-center justify-between gap-2 z-20">
        <div className="flex items-center justify-between w-full sm:w-auto gap-2">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 sm:p-2 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold transition-all shadow-lg shadow-cyan-600/20"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            <span className="text-xs font-mono text-cyan-300 shrink-0 font-bold">
              Step: {selectedTimeStep >= 0 ? `+${selectedTimeStep}h` : `${selectedTimeStep}h`}
            </span>
          </div>

          <div className="sm:hidden text-[10px] text-slate-400 font-mono">
            Pan-India NCUM Ensemble
          </div>
        </div>

        {/* Scrollable forecast steps with right padding for mobile chatbot button */}
        <div className="flex items-center gap-1 overflow-x-auto w-full sm:w-auto py-1 no-scrollbar pr-14 sm:pr-0">
          {timeSteps.map(step => (
            <button
              key={step.hour}
              onClick={() => setSelectedTimeStep(step.hour)}
              className={`px-2.5 py-1 rounded-md text-[11px] sm:text-xs font-mono font-medium transition-all shrink-0 ${
                selectedTimeStep === step.hour
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold shadow-lg shadow-cyan-500/25'
                  : 'bg-[#111827] text-slate-400 hover:text-slate-200 border border-[#1e2d48] hover:border-[#2a3f5f]'
              }`}
            >
              {step.label}
            </button>
          ))}
        </div>

        <div className="hidden lg:flex items-center gap-2 text-[11px] text-slate-500 shrink-0">
          <RefreshCw className="h-3.5 w-3.5 text-cyan-400 animate-spin" style={{ animationDuration: '4s' }} />
          <span>Pan-India NCUM Ensemble Cycle Active</span>
        </div>
      </div>
    </div>
  );
};
