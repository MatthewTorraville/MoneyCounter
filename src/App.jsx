import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Camera,
  DollarSign,
  RotateCcw,
  Image as ImageIcon,
  CheckCircle2,
  AlertCircle,
  X,
  Plus,
  Minus,
  Loader2,
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const CAD_DENOMINATIONS = [
  { id: 'bill_100', label: '$100 Bill', short: '$100', value: 100.0, type: 'bill', accent: 'bg-amber-500/15 text-amber-500 border-amber-500/30' },
  { id: 'bill_50',  label: '$50 Bill',  short: '$50',  value: 50.0,  type: 'bill', accent: 'bg-red-500/15 text-red-400 border-red-500/30' },
  { id: 'bill_20',  label: '$20 Bill',  short: '$20',  value: 20.0,  type: 'bill', accent: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
  { id: 'bill_10',  label: '$10 Bill',  short: '$10',  value: 10.0,  type: 'bill', accent: 'bg-purple-500/15 text-purple-400 border-purple-500/30' },
  { id: 'bill_5',   label: '$5 Bill',   short: '$5',   value: 5.0,   type: 'bill', accent: 'bg-blue-500/15 text-blue-400 border-blue-500/30' },
  { id: 'coin_2',   label: 'Toonie',    short: '$2',   value: 2.0,   type: 'coin', accent: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30' },
  { id: 'coin_1',   label: 'Loonie',    short: '$1',   value: 1.0,   type: 'coin', accent: 'bg-amber-400/15 text-amber-300 border-amber-400/30' },
  { id: 'coin_025', label: 'Quarter',   short: '25¢',  value: 0.25,  type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
  { id: 'coin_010', label: 'Dime',      short: '10¢',  value: 0.1,   type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
  { id: 'coin_005', label: 'Nickel',    short: '5¢',   value: 0.05,  type: 'coin', accent: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30' },
];

const INITIAL_COUNTS = CAD_DENOMINATIONS.reduce((acc, item) => {
  acc[item.id] = 0;
  return acc;
}, {});

export default function App() {
  // Theme state
  const [theme, setTheme] = useState('dark');
  const isDark = theme === 'dark';

  // TensorFlow / COCO-SSD model state (fixed missing state hooks from backend snippet)
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);

  // Media & Camera state
  const [imageDataUrl, setImageDataUrl] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState([]);

  // Currency counts state
  const [counts, setCounts] = useState(INITIAL_COUNTS);
  const [status, setStatus] = useState(null);

  // Refs
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const previewImgRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const requestRef = useRef(null);

  // Load TensorFlow.js & COCO-SSD on startup
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        if (!cancelled) {
          setModel(loadedModel);
          setIsModelLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setIsModelLoading(false);
          setStatus({ type: 'error', text: 'Failed to load TensorFlow COCO-SSD model.' });
        }
      }
    }
    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  // Draw bounding boxes on canvas (works for both uploaded images and live video)
  const drawDetectionsOnCanvas = useCallback((preds, sourceWidth, sourceHeight) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !sourceWidth || !sourceHeight) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width || sourceWidth;
    canvas.height = rect.height || sourceHeight;

    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    preds.forEach((pred) => {
      const [rawX, rawY, rawW, rawH] = pred.bbox;
      const x = rawX * scaleX;
      const y = rawY * scaleY;
      const w = rawW * scaleX;
      const h = rawH * scaleY;

      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      const labelText = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = '#16a34a';
      ctx.fillRect(x, Math.max(0, y - 20), textWidth + 10, 20);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, Math.max(14, y - 6));
    });
  }, []);

  // Redraw static image boxes on window resize
  useEffect(() => {
    if (isCameraOpen || !previewImgRef.current) return;
    const handleResize = () => {
      const img = previewImgRef.current;
      if (img && detections.length > 0) {
        drawDetectionsOnCanvas(detections, img.naturalWidth, img.naturalHeight);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [detections, imageDataUrl, isCameraOpen, drawDetectionsOnCanvas]);

  // Stop webcam & animation loop cleanly
  const stopCamera = useCallback(() => {
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
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

  // Real-time detection loop while camera is live
  useEffect(() => {
    if (!isCameraOpen || !model) return;
    let active = true;

    const detectFrame = async () => {
      if (!active) return;
      const video = videoRef.current;
      if (video && video.readyState === 4) {
        const preds = await model.detect(video);
        if (active) {
          setDetections(preds);
          drawDetectionsOnCanvas(preds, video.videoWidth, video.videoHeight);
        }
      }
      if (active) {
        requestRef.current = requestAnimationFrame(detectFrame);
      }
    };

    requestRef.current = requestAnimationFrame(detectFrame);
    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isCameraOpen, model, drawDetectionsOnCanvas]);

  // Load uploaded image file
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
      setStatus({ type: 'info', text: `Loaded "${file.name}". Click COUNT MONEY to run detection.` });
    };
    reader.readAsDataURL(file);
  };

  // Start live camera when user clicks "Use Camera"
  const handleStartCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus({ type: 'error', text: 'Camera access is not supported in this browser.' });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      setImageDataUrl(null);
      setFileName('');
      setDetections([]);
      setIsCameraOpen(true);
      setStatus({ type: 'info', text: 'Camera live. Live COCO-SSD detection is active—or click Capture Photo.' });
    } catch (err) {
      setStatus({ type: 'error', text: 'Camera permission denied or unavailable. Use Import Image instead.' });
    }
  };

  useEffect(() => {
    if (isCameraOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [isCameraOpen]);

  // Capture still photo from live video
  const handleCapturePhoto = () => {
    const video = videoRef.current;
    if (!video) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    const capturedDataUrl = tempCanvas.toDataURL('image/png');
    stopCamera();
    setImageDataUrl(capturedDataUrl);
    setFileName('camera-capture.png');
    setStatus({ type: 'info', text: 'Photo captured! Click COUNT MONEY to scan it.' });
  };

  // Run COCO-SSD detection on the uploaded/captured image
  const handleCountMoney = async () => {
    if (!model) {
      setStatus({ type: 'info', text: 'Model is still loading—wait a moment and try again.' });
      return;
    }
    if (!imageDataUrl && !isCameraOpen) {
      setStatus({ type: 'error', text: 'Please import an image or open the camera first.' });
      return;
    }

    setIsProcessing(true);
    try {
      const targetEl = isCameraOpen ? videoRef.current : previewImgRef.current;
      if (!targetEl) return;

      const preds = await model.detect(targetEl);
      setDetections(preds);

      if (!isCameraOpen) {
        drawDetectionsOnCanvas(preds, targetEl.naturalWidth, targetEl.naturalHeight);
      }

      const summary =
        preds.length > 0
          ? `Detected ${preds.length} object(s): ${preds.map((p) => `${p.class} (${(p.score * 100).toFixed(0)}%)`).join(', ')}`
          : 'No objects detected by COCO-SSD. (Note: standard COCO-SSD detects 80 general objects—swap in a custom CAD currency model to auto-fill counts!)';

      setStatus({ type: preds.length > 0 ? 'success' : 'info', text: summary });
    } catch (err) {
      setStatus({ type: 'error', text: 'Error running TensorFlow detection.' });
    } finally {
      setIsProcessing(false);
    }
  };

  const adjustCount = (id, delta) => {
    setCounts((prev) => ({
      ...prev,
      [id]: Math.max(0, (prev[id] || 0) + delta),
    }));
  };

  const handleReset = () => {
    setCounts(INITIAL_COUNTS);
    setDetections([]);
    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    setStatus({ type: 'info', text: 'Tally reset to $0.00 CAD.' });
  };

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
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&display=swap');
        .font-brand { font-family: 'Fredoka', system-ui, sans-serif; }
      `}</style>

      {/* TOP-CENTER GREEN THEME TOGGLE */}
      <header className="w-full flex justify-center">
        <button
          type="button"
          onClick={() => setTheme(isDark ? 'light' : 'dark')}
          className="bg-[#16a34a] hover:bg-[#22c55e] active:bg-[#15803d] text-white font-semibold text-sm px-5 py-2 rounded-b-xl shadow-md transition-all cursor-pointer tracking-wide"
        >
          {isDark ? 'Light Theme' : 'Dark Theme'}
        </button>
      </header>

      <main className="w-full max-w-4xl px-4 sm:px-6 pt-10 sm:pt-14 pb-16 flex flex-col items-center">
        {/* LOGO HEADER */}
        <div className="flex items-center justify-center gap-3 sm:gap-4 mb-4 select-none">
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

        {/* MODEL STATUS BADGE */}
        <div className="mb-8">
          <span
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
              isModelLoading
                ? 'border-amber-500/30 bg-amber-500/10 text-amber-400'
                : 'border-[#16a34a]/30 bg-[#16a34a]/10 text-[#22c55e]'
            }`}
          >
            {isModelLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading TensorFlow COCO-SSD model...
              </>
            ) : (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                COCO-SSD Vision Model Ready
              </>
            )}
          </span>
        </div>

        {/* ACTION BAR */}
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
            {/* Import Image Button */}
            <div className="flex-1 flex flex-col">
              <span className={`text-xs sm:text-sm font-medium mb-1.5 ${isDark ? 'text-zinc-200' : 'text-slate-700'}`}>
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

            {/* Use Camera Button */}
            <div className="flex flex-col md:w-44">
              <span className={`text-xs sm:text-sm font-medium mb-1.5 ${isDark ? 'text-zinc-200' : 'text-slate-700'}`}>
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

            {/* Primary Green COUNT MONEY Button */}
            <div className="flex flex-col">
              <span className="hidden md:block text-xs sm:text-sm font-medium text-transparent select-none mb-1.5">
                Count
              </span>
              <button
                type="button"
                onClick={handleCountMoney}
                disabled={isProcessing || isModelLoading}
                className="h-12 px-6 rounded-lg bg-[#16a34a] hover:bg-[#22c55e] active:bg-[#15803d] disabled:opacity-60 text-white font-bold text-sm sm:text-base tracking-wide uppercase flex items-center justify-center gap-2 shadow-md shadow-[#16a34a]/25 transition cursor-pointer shrink-0"
              >
                <DollarSign className="w-5 h-5 stroke-[2.5]" />
                <span>{isProcessing ? 'SCANNING...' : 'COUNT MONEY'}</span>
              </button>
            </div>
          </div>

          {/* Status Banner */}
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

        {/* PREVIEW CANVAS & CURRENCY BREAKDOWN */}
        <section className="w-full max-w-3xl mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Image / Camera Preview Card */}
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
                if (e.dataTransfer.files?.[0]) loadImageFile(e.dataTransfer.files[0]);
              }}
              className={`relative w-full min-h-[290px] sm:min-h-[350px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center overflow-hidden transition-colors ${
                isDragging
                  ? 'border-[#22c55e] bg-[#16a34a]/10'
                  : isDark
                  ? 'bg-[#0d0d0d] border-zinc-800 hover:border-[#16a34a]/70'
                  : 'bg-white border-slate-300 hover:border-[#16a34a]'
              }`}
            >
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

              {isCameraOpen && (
                <div className="relative flex items-center justify-center w-full bg-black">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full max-h-[400px] object-contain block"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="pointer-events-none absolute inset-0 w-full h-full"
                  />
                </div>
              )}

              {!isCameraOpen && imageDataUrl && (
                <div className="relative flex items-center justify-center w-full">
                  <img
                    ref={previewImgRef}
                    src={imageDataUrl}
                    alt="Currency preview"
                    className="max-w-full max-h-[400px] w-auto h-auto object-contain block"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="pointer-events-none absolute inset-0 w-full h-full"
                  />
                </div>
              )}
            </div>

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

          {/* Canadian Currency Breakdown Panel */}
          <div
            className={`lg:col-span-5 rounded-xl border p-5 ${
              isDark ? 'bg-[#0d0d0d] border-zinc-800' : 'bg-white border-slate-200 shadow-sm'
            }`}
          >
            <div
              className={`flex items-center justify-between pb-3.5 border-b ${
                isDark ? 'border-zinc-800' : 'border-slate-200'
              }`}
            >
              <div>
                <span className={`text-xs font-bold uppercase tracking-wider ${isDark ? 'text-zinc-400' : 'text-slate-500'}`}>
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

            {/* CAD Bills */}
            <div className="mt-4">
              <div className={`flex items-center justify-between text-xs font-semibold uppercase tracking-wider mb-2 ${isDark ? 'text-zinc-400' : 'text-slate-500'}`}>
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
                      <span className={`px-2 py-0.5 rounded border text-xs font-bold tabular-nums ${item.accent}`}>
                        {item.short}
                      </span>
                      <span className="font-medium">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, -1)}
                        className={`w-6 h-6 rounded flex items-center justify-center cursor-pointer transition ${
                          isDark ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300' : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
                        }`}
                      >
                        <Minus className="w-3.5 h-3.5" />
                      </button>
                      <span className="w-6 text-center font-bold tabular-nums">{counts[item.id] || 0}</span>
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, 1)}
                        className="w-6 h-6 rounded bg-[#16a34a]/20 hover:bg-[#16a34a] text-[#16a34a] hover:text-white flex items-center justify-center cursor-pointer transition"
                      >
                        <Plus className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* CAD Coins */}
            <div className={`mt-5 pt-4 border-t ${isDark ? 'border-zinc-800' : 'border-slate-200'}`}>
              <div className={`flex items-center justify-between text-xs font-semibold uppercase tracking-wider mb-2 ${isDark ? 'text-zinc-400' : 'text-slate-500'}`}>
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
                      <span className={`px-2 py-0.5 rounded border text-xs font-bold tabular-nums ${item.accent}`}>
                        {item.short}
                      </span>
                      <span className="font-medium">{item.label}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, -1)}
                        className={`w-6 h-6 rounded flex items-center justify-center cursor-pointer transition ${
                          isDark ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300' : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
                        }`}
                      >
                        <Minus className="w-3.5 h-3.5" />
                      </button>
                      <span className="w-6 text-center font-bold tabular-nums">{counts[item.id] || 0}</span>
                      <button
                        type="button"
                        onClick={() => adjustCount(item.id, 1)}
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