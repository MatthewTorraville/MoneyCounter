import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, Camera, DollarSign, RotateCcw, Image as ImageIcon, CheckCircle2, AlertCircle, X, Plus, Minus } from 'lucide-react';

/**
 * Canadian Currency Denominations Configuration
 */
const CAD_DENOMINATIONS = [
  { id: 'bill_100', label: '$100 Bill', short: '$100', value: 100.00, type: 'bill', accent: 'bg-amber-500/15 text-amber-500 border-amber-500/30' },
  { id: 'bill_50',  label: '$50 Bill',  short: '$50',  value: 50.00,  type: 'bill', accent: 'bg-red-500/15 text-red-400 border-red-500/30' },
  { id: 'bill_20',  label: '$20 Bill',  short: '$20',  value: 20.00,  type: 'bill', accent: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
  { id: 'bill_10',  label: '$10 Bill',  short: '$10',  value: 10.00,  type: 'bill', accent: 'bg-purple-500/15 text-purple-400 border-purple-500/30' },
  { id: 'bill_5',   label: '$5 Bill',   short: '$5',   value: 5.00,   type: 'bill', accent: 'bg-blue-500/15 text-blue-400 border-blue-500/30' },
  { id: 'coin_2',   label: 'Toonie',    short: '$2',   value: 2.00,   type: 'coin', accent: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30' },
  { id: 'coin_1',   label: 'Loonie',    short: '$1',   value: 1.00,   type: 'coin', accent: 'bg-amber-400/15 text-amber-300 border-amber-400/30' },
  { id: 'coin_025', label: 'Quarter',   short: '25¢',  value: 0.25,   type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
  { id: 'coin_010', label: 'Dime',      short: '10¢',  value: 0.10,   type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
  { id: 'coin_005', label: 'Nickel',    short: '5¢',   value: 0.05,   type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
];

const INITIAL_COUNTS = CAD_DENOMINATIONS.reduce((acc, item) => {
  acc[item.id] = 0;
  return acc;
}, {});

/**
 * ============================================================================
 * COMPUTER VISION / DETECTION HOOK
 * Plug your custom ML/CV inference (TensorFlow.js, ONNX, Roboflow, API) here.
 * ============================================================================
 * @param {string} imageDataUrl - Base64 data URL of the uploaded or captured photo
 * @returns {Promise<{ counts?: Record<string, number>, detections?: Array<{ label: string, confidence?: number, bbox: number[] }>, message?: string }>}
 */
export async function processMoneyImage(imageDataUrl) {
  // Simulate brief processing delay so UI feedback is clear
  await new Promise((resolve) => setTimeout(resolve, 450));

  // TODO: Replace this placeholder return with your model inference on `imageDataUrl`
  // Example return format:
  // return {
  //   counts: { bill_20: 2, bill_5: 1, coin_2: 3 },
  //   detections: [
  //     { label: '$20 Bill', confidence: 0.97, bbox: [0.12, 0.18, 0.42, 0.28] }
  //   ],
  //   message: 'Detected 6 Canadian currency items.'
  // };
  return {
    counts: null,
    detections: [],
    message: 'Image ready. Connect your CV model inside processMoneyImage() or adjust counts manually.',
  };
}

export default function App() {
  // Theme state ('dark' defaults to pitch-black #050505 matching reference screenshot)
  const [theme, setTheme] = useState('dark');
  const isDark = theme === 'dark';

  // Media & Camera state
  const [imageDataUrl, setImageDataUrl] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState([]);

  // Currency counts state
  const [counts, setCounts] = useState(INITIAL_COUNTS);

  // Non-blocking status banner
  const [status, setStatus] = useState(null); // { type: 'info' | 'success' | 'error', text: string }

  // Refs
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const previewImgRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  // Stop webcam cleanly on unmount
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOpen(false);
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  // Draw bounding boxes whenever detections or image size change
  const renderBoundingBoxes = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const img = previewImgRef.current;
    if (!canvas || !img) return;

    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      let [x, y, w, h] = det.bbox;
      if (w <= 1 && h <= 1) {
        x *= canvas.width;
        y *= canvas.height;
        w *= canvas.width;
        h *= canvas.height;
      }

      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      const labelText = `${det.label}${det.confidence ? ` ${(det.confidence * 100).toFixed(0)}%` : ''}`;
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = '#16a34a';
      ctx.fillRect(x, Math.max(0, y - 20), textWidth + 10, 20);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, Math.max(14, y - 6));
    });
  }, [detections]);

  useEffect(() => {
    renderBoundingBoxes();
    window.addEventListener('resize', renderBoundingBoxes);
    return () => window.removeEventListener('resize', renderBoundingBoxes);
  }, [renderBoundingBoxes, imageDataUrl]);

  // Load an image file into base64 data URL
  const loadImageFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setStatus({ type: 'error', text: 'Please select a valid image file (PNG, JPG, WEBP).' });
      return;
    }
    stopCamera();
    setFileName(file.name);
    setDetections([]);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImageDataUrl(e.target.result);
      setStatus({ type: 'info', text: `Loaded "${file.name}". Click COUNT MONEY to analyze.` });
    };
    reader.readAsDataURL(file);
  };

  // Start live device camera
  const handleStartCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus({ type: 'error', text: 'Camera access is not supported in this browser.' });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      streamRef.current = stream;
      setImageDataUrl(null);
      setFileName('');
      setDetections([]);
      setIsCameraOpen(true);
      setStatus({ type: 'info', text: 'Camera live. Frame your bills and coins, then click Capture Photo.' });
    } catch (err) {
      setStatus({ type: 'error', text: 'Camera permission denied or unavailable. Use Import Image instead.' });
    }
  };

  // Attach MediaStream once video element mounts
  useEffect(() => {
    if (isCameraOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [isCameraOpen]);

  // Capture photo from live video stream
  const handleCapturePhoto = () => {
    const video = videoRef.current;
    if (!video) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth || 1280;
    tempCanvas.height = video.videoHeight || 720;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    const capturedDataUrl = tempCanvas.toDataURL('image/png');
    stopCamera();
    setImageDataUrl(capturedDataUrl);
    setFileName('camera-capture.png');
    setStatus({ type: 'info', text: 'Photo captured! Click COUNT MONEY to run currency detection.' });
  };

  // Run the processMoneyImage(imageDataUrl) hook
  const handleCountMoney = async () => {
    if (!imageDataUrl) {
      setStatus({ type: 'error', text: 'Please import an image or capture a photo first.' });
      return;
    }

    setIsProcessing(true);
    try {
      const result = await processMoneyImage(imageDataUrl);
      if (result?.counts) {
        setCounts((prev) => {
          const next = { ...prev };
          Object.entries(result.counts).forEach(([key, qty]) => {
            if (key in next) next[key] = qty;
          });
          return next;
        });
      }
      if (Array.isArray(result?.detections)) {
        setDetections(result.detections);
      }
      setStatus({
        type: 'success',
        text: result?.message || 'Currency scan complete.',
      });
    } catch (err) {
      setStatus({ type: 'error', text: 'Error processing currency image.' });
    } finally {
      setIsProcessing(false);
    }
  };

  // Adjust individual denomination counts manually
  const adjustCount = (id, delta) => {
    setCounts((prev) => ({
      ...prev,
      [id]: Math.max(0, (prev[id] || 0) + delta),
    }));
  };

  const handleReset = () => {
    setCounts(INITIAL_COUNTS);
    setDetections([]);
    setStatus({ type: 'info', text: 'Tally reset to $0.00 CAD.' });
  };

  // Calculate subtotals & grand total
  const billsSubtotal = CAD_DENOMINATIONS.filter((d) => d.type === 'bill').reduce(
    (sum, d) => sum + (counts[d.id] || 0) * d.value,
    0
  );
  const coinsSubtotal = CAD_DENOMINATIONS.filter((d) => d.type === 'coin').reduce(
    (sum, d) => sum + (counts[d.id] || 0) * d.value,
    0
  );
  const grandTotal = billsSubtotal + coinsSubtotal;

  return (
    <div
      className={`min-h-screen transition-colors duration-200 font-sans flex flex-col items-center ${
        isDark ? 'dark bg-[#050505] text-white' : 'light bg-[#f8fafc] text-slate-900'
      }`}
    >
      {/* Inline font import for bold rounded logotype matching CnvMP3 reference */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&display=swap');
        .font-brand { font-family: 'Fredoka', system-ui, sans-serif; }
      `}</style>

      {/* ===================================================================== */}
      {/* 1. TOP-CENTER PILL BUTTON FOR DARK / LIGHT THEME                      */}
      {/* ===================================================================== */}
      <header className="w-full flex justify-center">
        <button
          id="top-theme-pill"
          type="button"
          onClick={() => setTheme(isDark ? 'light' : 'dark')}
          className="bg-[#16a34a] hover:bg-[#22c55e] active:bg-[#15803d] text-white font-semibold text-sm px-5 py-2 rounded-b-xl shadow-md transition-all cursor-pointer tracking-wide"
        >
          {isDark ? 'Light Theme' : 'Dark Theme'}
        </button>
      </header>

      {/* ===================================================================== */}
      {/* 2. MAIN CONTAINER                                                     */}
      {/* ===================================================================== */}
      <main className="w-full max-w-4xl px-4 sm:px-6 pt-10 sm:pt-14 pb-16 flex flex-col items-center">
        {/* Centered Brand Header (Inline SVG circular currency icon + two-tone logotype) */}
        <div className="flex items-center justify-center gap-3 sm:gap-4 mb-10 sm:mb-12 select-none">
          <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-full bg-[#16a34a] flex items-center justify-center shadow-lg shadow-[#16a34a]/25 shrink-0">
            <svg
              className="w-7 h-7 sm:w-8 sm:h-8 text-[#050505]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {/* Circular refresh arrows + central dollar sign */}
              <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
              <path d="M21 3v5h-5" />
              <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
              <path d="M8 16H3v5" />
              <path d="M12 7.5v9" />
              <path d="M14.5 9.5a2 2 0 0 0-2-1.5h-1a2 2 0 0 0 0 4h1a2 2 0 0 1 0 4h-1a2 2 0 0 1-2-1.5" />
            </svg>
          </div>

          <h1 className="font-brand text-3xl sm:text-5xl font-bold tracking-tight flex items-center">
            <span className={isDark ? 'text-white' : 'text-slate-900'}>MONEY</span>
            <span className="text-[#16a34a]">COUNTER</span>
          </h1>
        </div>

        {/* =================================================================== */}
        {/* 3. MAIN ACTION BAR (Import Image | Use Camera | COUNT MONEY)        */}
        {/* =================================================================== */}
        <section aria-label="Currency Input Actions" className="w-full max-w-3xl">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) loadImageFile(e.target.files[0]);
              e.target.value = '';
            }}
          />

          <div className="flex flex-col md:flex-row items-stretch md:items-end justify-center gap-3 sm:gap-4">
            {/* 1. Import Image Button (styled like the white input box in CnvMP3) */}
            <div className="flex-1 flex flex-col">
              <span
                className={`text-xs sm:text-sm font-medium mb-1.5 ${
                  isDark ? 'text-zinc-200' : 'text-slate-700'
                }`}
              >
                Import Currency Photo
              </span>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className={`h-12 px-4 rounded-lg font-medium text-sm sm:text-base flex items-center justify-between gap-3 border transition cursor-pointer ${
                  isDark
                    ? 'bg-white text-zinc-800 border-transparent hover:bg-zinc-100'
                    : 'bg-white text-slate-800 border-slate-300 hover:bg-slate-50 shadow-sm'
                }`}
              >
                <span className={`truncate ${fileName ? 'text-zinc-900 font-semibold' : 'text-zinc-500'}`}>
                  {fileName || 'Select image or drop below...'}
                </span>
                <span className="inline-flex items-center gap-1.5 text-[#16a34a] font-semibold shrink-0">
                  <Upload className="w-4 h-4" />
                  Import Image
                </span>
              </button>
            </div>

            {/* 2. Use Camera Button (styled like the outlined dropdown box in CnvMP3) */}
            <div className="flex flex-col md:w-44">
              <span
                className={`text-xs sm:text-sm font-medium mb-1.5 ${
                  isDark ? 'text-zinc-200' : 'text-slate-700'
                }`}
              >
                Live Capture
              </span>
              <button
                type="button"
                onClick={isCameraOpen ? stopCamera : handleStartCamera}
                className={`h-12 px-4 rounded-lg border font-medium text-sm sm:text-base flex items-center justify-center gap-2 transition cursor-pointer ${
                  isCameraOpen
                    ? 'bg-red-600/15 border-red-500 text-red-500 hover:bg-red-600/25'
                    : isDark
                    ? 'bg-[#0a0a0a] border-zinc-700 text-white hover:border-[#22c55e]'
                    : 'bg-white border-slate-300 text-slate-900 hover:border-[#16a34a] shadow-sm'
                }`}
              >
                <Camera className="w-4 h-4 text-[#16a34a]" />
                <span>{isCameraOpen ? 'Stop Camera' : 'Use Camera'}</span>
              </button>
            </div>

            {/* 3. Primary Green COUNT MONEY Button (styled like CONVERT in CnvMP3) */}
            <div className="flex flex-col">
              <span className="hidden md:block text-xs sm:text-sm font-medium text-transparent select-none mb-1.5">
                Count
              </span>
              <button
                type="button"
                onClick={handleCountMoney}
                disabled={isProcessing}
                className="h-12 px-6 rounded-lg bg-[#16a34a] hover:bg-[#22c55e] active:bg-[#15803d] disabled:opacity-60 text-white font-bold text-sm sm:text-base tracking-wide uppercase flex items-center justify-center gap-2 shadow-md shadow-[#16a34a]/25 transition cursor-pointer shrink-0"
              >
                <DollarSign className="w-5 h-5 stroke-[2.5]" />
                <span>{isProcessing ? 'COUNTING...' : 'COUNT MONEY'}</span>
              </button>
            </div>
          </div>

          {/* Inline Status Banner */}
          {status && (
            <div
              role="status"
              className={`mt-4 px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center justify-between gap-3 ${
                status.type === 'error'
                  ? 'bg-red-500/10 border-red-500/40 text-red-500'
                  : status.type === 'success'
                  ? 'bg-[#16a34a]/15 border-[#16a34a]/40 text-[#22c55e]'
                  : isDark
                  ? 'bg-zinc-900 border-zinc-800 text-zinc-300'
                  : 'bg-white border-slate-200 text-slate-700'
              }`}
            >
              <div className="flex items-center gap-2">
                {status.type === 'error' ? (
                  <AlertCircle className="w-4 h-4 shrink-0" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-[#16a34a]" />
                )}
                <span>{status.text}</span>
              </div>
              <button
                type="button"
                onClick={() => setStatus(null)}
                aria-label="Dismiss message"
                className="opacity-70 hover:opacity-100 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </section>

        {/* =================================================================== */}
        {/* 4. PREVIEW CANVAS & CANADIAN CURRENCY BREAKDOWN PANEL               */}
        {/* =================================================================== */}
        <section className="w-full max-w-3xl mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Image / Camera Preview Canvas Card (7 columns) */}
          <div className="lg:col-span-7 flex flex-col">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                if (e.dataTransfer.files?.[0]) {
                  loadImageFile(e.dataTransfer.files[0]);
                }
              }}
              className={`relative w-full min-h-[290px] sm:min-h-[350px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center overflow-hidden transition-colors ${
                isDragging
                  ? 'border-[#22c55e] bg-[#16a34a]/10'
                  : isDark
                  ? 'bg-[#0d0d0d] border-zinc-800 hover:border-[#16a34a]/70'
                  : 'bg-white border-slate-300 hover:border-[#16a34a]'
              }`}
            >
              {/* Empty Dropzone State */}
              {!isCameraOpen && !imageDataUrl && (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="p-6 text-center flex flex-col items-center justify-center cursor-pointer select-none"
                >
                  <div className="w-12 h-12 rounded-full bg-[#16a34a]/15 text-[#16a34a] flex items-center justify-center mb-3">
                    <ImageIcon className="w-6 h-6" />
                  </div>
                  <p className="text-sm sm:text-base font-semibold">
                    Drag &amp; drop a photo here, or click to browse
                  </p>
                  <p className={`text-xs mt-1 ${isDark ? 'text-zinc-400' : 'text-slate-500'}`}>
                    Supports Canadian bills ($5–$100) and coins (5¢–$2)
                  </p>
                </div>
              )}

              {/* Live Camera Feed */}
              {isCameraOpen && (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full max-h-[400px] object-contain bg-black"
                />
              )}

              {/* Uploaded / Captured Image + Detection Overlay Canvas */}
              {!isCameraOpen && imageDataUrl && (
                <div className="relative flex items-center justify-center w-full">
                  <img
                    ref={previewImgRef}
                    src={imageDataUrl}
                    alt="Currency preview"
                    onLoad={renderBoundingBoxes}
                    className="max-w-full max-h-[400px] w-auto h-auto object-contain block"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="pointer-events-none absolute inset-0 m-auto"
                  />
                </div>
              )}
            </div>

            {/* Camera Controls (Capture Photo / Cancel) or Clear Image */}
            {isCameraOpen && (
              <div className="mt-3 flex items-center justify-center gap-3">
                <button
                  type="button"
                  onClick={handleCapturePhoto}
                  className="px-5 py-2.5 rounded-lg bg-[#16a34a] hover:bg-[#22c55e] text-white font-semibold text-sm flex items-center gap-2 shadow-sm cursor-pointer transition"
                >
                  <Camera className="w-4 h-4" />
                  Capture Photo
                </button>
                <button
                  type="button"
                  onClick={stopCamera}
                  className={`px-4 py-2.5 rounded-lg font-semibold text-sm border cursor-pointer transition ${
                    isDark
                      ? 'bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800'
                      : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  Cancel
                </button>
              </div>
            )}

            {!isCameraOpen && imageDataUrl && (
              <div className="mt-3 flex items-center justify-between">
                <span className={`text-xs truncate max-w-[220px] ${isDark ? 'text-zinc-400' : 'text-slate-500'}`}>
                  {fileName}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setImageDataUrl(null);
                    setFileName('');
                    setDetections([]);
                  }}
                  className="text-xs font-medium text-red-500 hover:text-red-400 cursor-pointer"
                >
                  Remove Photo
                </button>
              </div>
            )}
          </div>

          {/* Canadian Currency Breakdown Tally Panel (5 columns) */}
          <div
            className={`lg:col-span-5 rounded-xl border p-5 ${
              isDark ? 'bg-[#0d0d0d] border-zinc-800' : 'bg-white border-slate-200 shadow-sm'
            }`}
          >
            {/* Total CAD Header */}
            <div
              className={`flex items-center justify-between pb-3.5 border-b ${
                isDark ? 'border-zinc-800' : 'border-slate-200'
              }`}
            >
              <div>
                <span
                  className={`text-xs font-bold uppercase tracking-wider ${
                    isDark ? 'text-zinc-400' : 'text-slate-500'
                  }`}
                >
                  Total CAD
                </span>
                <div className="text-3xl sm:text-4xl font-extrabold text-[#16a34a] tabular-nums mt-0.5">
                  ${grandTotal.toFixed(2)}
                </div>
              </div>

              <button
                type="button"
                onClick={handleReset}
                className={`px-3 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition ${
                  isDark
                    ? 'bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800'
                    : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'
                }`}
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            </div>

            {/* CAD Banknotes ($5, $10, $20, $50, $100) */}
            <div className="mt-4">
              <div
                className={`flex items-center justify-between text-xs font-semibold uppercase tracking-wider mb-2 ${
                  isDark ? 'text-zinc-400' : 'text-slate-500'
                }`}
              >
                <span>CAD Bills</span>
                <span className="tabular-nums">${billsSubtotal.toFixed(2)}</span>
              </div>

              <div className="space-y-1.5">
                {CAD_DENOMINATIONS.filter((d) => d.type === 'bill').map((item) => (
                  <div
                    key={item.id}
                    className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-sm ${
                      isDark ? 'bg-zinc-900/80' : 'bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded border text-xs font-bold tabular-nums ${item.accent}`}
                      >
                        {item.short}
                      </span>
                      <span className="font-medium">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, -1)}
                        aria-label={`Decrease ${item.label}`}
                        className={`w-6 h-6 rounded flex items-center justify-center cursor-pointer transition ${
                          isDark
                            ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300'
                            : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
                        }`}
                      >
                        <Minus className="w-3.5 h-3.5" />
                      </button>
                      <span className="w-6 text-center font-bold tabular-nums">
                        {counts[item.id] || 0}
                      </span>
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, 1)}
                        aria-label={`Increase ${item.label}`}
                        className="w-6 h-6 rounded bg-[#16a34a]/20 hover:bg-[#16a34a] text-[#16a34a] hover:text-white flex items-center justify-center cursor-pointer transition"
                      >
                        <Plus className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* CAD Coins ($0.05, $0.10, $0.25, $1, $2) */}
            <div
              className={`mt-5 pt-4 border-t ${
                isDark ? 'border-zinc-800' : 'border-slate-200'
              }`}
            >
              <div
                className={`flex items-center justify-between text-xs font-semibold uppercase tracking-wider mb-2 ${
                  isDark ? 'text-zinc-400' : 'text-slate-500'
                }`}
              >
                <span>CAD Coins</span>
                <span className="tabular-nums">${coinsSubtotal.toFixed(2)}</span>
              </div>

              <div className="space-y-1.5">
                {CAD_DENOMINATIONS.filter((d) => d.type === 'coin').map((item) => (
                  <div
                    key={item.id}
                    className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-sm ${
                      isDark ? 'bg-zinc-900/80' : 'bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded border text-xs font-bold tabular-nums ${item.accent}`}
                      >
                        {item.short}
                      </span>
                      <span className="font-medium">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, -1)}
                        aria-label={`Decrease ${item.label}`}
                        className={`w-6 h-6 rounded flex items-center justify-center cursor-pointer transition ${
                          isDark
                            ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300'
                            : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
                        }`}
                      >
                        <Minus className="w-3.5 h-3.5" />
                      </button>
                      <span className="w-6 text-center font-bold tabular-nums">
                        {counts[item.id] || 0}
                      </span>
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, 1)}
                        aria-label={`Increase ${item.label}`}
                        className="w-6 h-6 rounded bg-[#16a34a]/20 hover:bg-[#16a34a] text-[#16a34a] hover:text-white flex items-center justify-center cursor-pointer transition"
                      >
                        <Plus className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}