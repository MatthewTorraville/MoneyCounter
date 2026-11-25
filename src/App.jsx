import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tflite from "@tensorflow/tfjs-tflite";

//Yes we shoved this whole project into one "small" dinky file
const MODEL_URL = "/models/cad_money.tflite";
const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.30;
const IOU_THRESHOLD = 0.45;

const DENOMINATIONS = [
  { id: "bill_100", name: "$100 bill", value: 100, type: "Bill" },
  { id: "bill_50", name: "$50 bill", value: 50, type: "Bill" },
  { id: "bill_20", name: "$20 bill", value: 20, type: "Bill" },
  { id: "bill_10", name: "$10 bill", value: 10, type: "Bill" },
  { id: "bill_5", name: "$5 bill", value: 5, type: "Bill" },
  { id: "coin_2", name: "Toonie", value: 2, type: "Coin" },
  { id: "coin_1", name: "Loonie", value: 1, type: "Coin" },
  { id: "coin_025", name: "Quarter", value: 0.25, type: "Coin" },
  { id: "coin_010", name: "Dime", value: 0.1, type: "Coin" },
  { id: "coin_005", name: "Nickel", value: 0.05, type: "Coin" },
];

const CLASS_NAMES = [
  "$1 coin", "10¢ coin", "$10 bill", "$100 bill", "$2 coin",
  "$20 bill", "25¢ coin", "5¢ coin", "$5 bill", "$50 bill",
];

const CLASS_TO_ID = [
  "coin_1", "coin_010", "bill_10", "bill_100", "coin_2",
  "bill_20", "coin_025", "coin_005", "bill_5", "bill_50",
];

const emptyCounts = () =>
  Object.fromEntries(DENOMINATIONS.map(({ id }) => [id, 0]));

let modelPromise;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      tflite.setWasmPath(
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/"
      );

      await tf.ready();
      const model = await tflite.loadTFLiteModel(MODEL_URL);
      const shape = model.inputs?.[0]?.shape;

      if (!shape || shape.length !== 4 || shape[0] !== 1) {
        throw new Error(`Weird lookin shape: ${JSON.stringify(shape)}`);
      }

      if (shape[1] === 3 && shape[2] === INPUT_SIZE && shape[3] === INPUT_SIZE) {
        model.inputLayout = "NCHW";
      } else if (
        shape[1] === INPUT_SIZE &&
        shape[2] === INPUT_SIZE &&
        shape[3] === 3
      ) {
        model.inputLayout = "NHWC";
      } else {
        throw new Error(`Model expects ${JSON.stringify(shape)}, but you got the wrong inpud bud`);
      }

      console.log("Model input:", shape);
      console.log("Model output:", model.outputs);
      return model;
    })().catch((error) => {
      modelPromise = null;
      throw error;
    });
  }

  return modelPromise;
}

function prepareImage(source) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;

  if (!width || !height) throw new Error("Getin' the image ready...");

  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;

  const context = canvas.getContext("2d");
  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const padX = (INPUT_SIZE - resizedWidth) / 2;
  const padY = (INPUT_SIZE - resizedHeight) / 2;

  context.drawImage(
    source, 0, 0, width, height,
    padX, padY, resizedWidth, resizedHeight
  );

  return {
    canvas,
    width,
    height,
    scale,
    padX,
    padY,
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return overlap / (areaA + areaB - overlap || 1);
}

function suppressDuplicates(detections) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];

  while (sorted.length && kept.length < 100) {
    const current = sorted.shift();
    kept.push(current);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (
        sorted[i].classId === current.classId &&
        iou(sorted[i], current) > IOU_THRESHOLD
      ) {
        sorted.splice(i, 1);
      }
    }
  }

  return kept;
}

function decodeOutput(tensor, image) {
  const [batch, dim1, dim2] = tensor.shape;
  if (batch !== 1 || tensor.shape.length !== 3) {
    throw new Error(`Unsupported output: ${JSON.stringify(tensor.shape)}`);
  }

  const channelsFirst = dim1 === 14;
  const channelsLast = dim2 === 14;
  const sixColumns = dim2 === 6;

  if (!channelsFirst && !channelsLast && !sixColumns) {
    throw new Error(`Unsupported output: ${JSON.stringify(tensor.shape)}. Expected sumin else`);
  }

  const data = tensor.dataSync();
  const candidateCount = channelsFirst ? dim2 : dim1;

  const read = (candidate, channel) =>
    channelsFirst
      ? data[channel * candidateCount + candidate]
      : data[candidate * dim2 + channel];

  const detections = [];

  for (let i = 0; i < candidateCount; i++) {
    let x1, y1, x2, y2, score, classId;

    if (sixColumns) {
      // rows are [x1, y1, x2, y2, confidence, classId].
      x1 = read(i, 0);
      y1 = read(i, 1);
      x2 = read(i, 2);
      y2 = read(i, 3);
      score = read(i, 4);
      classId = Math.round(read(i, 5));
    } else {
      // rows are [centerX, centerY, width, height, 10 class scores].
      const centerX = read(i, 0);
      const centerY = read(i, 1);
      const boxWidth = read(i, 2);
      const boxHeight = read(i, 3);

      score = -Infinity;
      classId = -1;

      for (let c = 0; c < CLASS_NAMES.length; c++) {
        const classScore = read(i, c + 4);
        if (classScore > score) {
          score = classScore;
          classId = c;
        }
      }

      x1 = centerX - boxWidth / 2;
      y1 = centerY - boxHeight / 2;
      x2 = centerX + boxWidth / 2;
      y2 = centerY + boxHeight / 2;
    }

    if (
      !Number.isFinite(score) ||
      score < SCORE_THRESHOLD ||
      classId < 0 ||
      classId >= CLASS_TO_ID.length
    ) continue;

    // Convert normalized coordinates to model pixels when necessary.
    if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5) {
      x1 *= INPUT_SIZE;
      y1 *= INPUT_SIZE;
      x2 *= INPUT_SIZE;
      y2 *= INPUT_SIZE;
    }

    // Remove the letterbox padding and stuff
    const left = Math.max(0, Math.min(image.width, (x1 - image.padX) / image.scale));
    const top = Math.max(0, Math.min(image.height, (y1 - image.padY) / image.scale));
    const right = Math.max(0, Math.min(image.width, (x2 - image.padX) / image.scale));
    const bottom = Math.max(0, Math.min(image.height, (y2 - image.padY) / image.scale));

    if (right <= left || bottom <= top) continue;

    detections.push({
      classId,
      currencyId: CLASS_TO_ID[classId],
      className: CLASS_NAMES[classId],
      score,
      box: { left, top, width: right - left, height: bottom - top },
      x1: left,
      y1: top,
      x2: right,
      y2: bottom,
    });
  }

  return suppressDuplicates(detections);
}

function findOutputTensor(value) {
  if (value && Array.isArray(value.shape) && typeof value.dataSync === "function") {
    return value;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      const tensor = findOutputTensor(item);
      if (tensor) return tensor;
    }
  }

  if (value && typeof value === "object") {
    for (const item of Object.values(value)) {
      const tensor = findOutputTensor(item);
      if (tensor) return tensor;
    }
  }

  return null;
}

function disposeOutput(value) {
  if (!value) return;

  if (typeof value.dispose === "function") {
    value.dispose();
    return;
  }

  if (Array.isArray(value)) {
    value.forEach(disposeOutput);
    return;
  }

  if (typeof value === "object") {
    Object.values(value).forEach(disposeOutput);
  }
}

async function detectMoney(model, source) {
  const image = prepareImage(source);
  let input;
  let output;
  let adjustedOutput;

  try {
    input = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(image.canvas).toFloat().div(255).expandDims(0);
      return model.inputLayout === "NCHW"
        ? pixels.transpose([0, 3, 1, 2])
        : pixels;
    });

    output = model.predict(input);
    let result = findOutputTensor(output);

    if (!result) {
      const outputDescription = output && typeof output === "object"
        ? Object.keys(output).join(", ")
        : String(output);
      throw new Error(
        `. Uh oh... Im cooked.: ${outputDescription}`
      );
    }

    // Math mmm
    while (result.shape.length > 3) {
      const axis = result.shape.findIndex((size, index) => index > 0 && size === 1);
      if (axis === -1) {
        throw new Error(`Unsupported model output shape: ${JSON.stringify(result.shape)}`);
      }

      const next = tf.squeeze(result, [axis]);
      if (adjustedOutput) adjustedOutput.dispose();
      adjustedOutput = next;
      result = adjustedOutput;
    }

    console.log("Model output shape:", result.shape);
    return decodeOutput(result, image);
  } finally {
    input?.dispose();
    adjustedOutput?.dispose();
    disposeOutput(output);
  }
}

export default function App() {
  const [model, setModel] = useState(null);
  const [modelStatus, setModelStatus] = useState("Loading model…");
  const [photo, setPhoto] = useState(null);
  const [fileName, setFileName] = useState("");
  const [counts, setCounts] = useState(emptyCounts);
  const [detections, setDetections] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);

  const imageRef = useRef(null);
  const fileRef = useRef(null);
  const captureRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    loadModel()
      .then((loaded) => {
        if (!cancelled) {
          setModel(loaded);
          setModelStatus("We're ready now");
        }
      })
      .catch((error) => {
        console.error(error);
        if (!cancelled) setModelStatus(`Uh oh: ${error.message}`);
      });

    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }

  async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      captureRef.current?.click();
      return;
    }

    setMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOpen(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      });
    } catch {
      captureRef.current?.click();
    }
  }

  function snapPhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      stopCamera();
      if (!blob) return;
      const snappedFile = new File([blob], "camera-photo.jpg", { type: "image/jpeg" });
      choosePhoto(snappedFile);
    }, "image/jpeg", 0.92);
  }

  function choosePhoto(file) {
    if (!file || !file.type.startsWith("image/")) {
      setMessage("Choose an image of some kind.");
      return;
    }

    setPhoto(URL.createObjectURL(file));
    setFileName(file.name);
    setDetections([]);
    setCounts(emptyCounts());
    setMessage("");
  }

  async function countMoney() {
    if (!model || !imageRef.current) {
      setMessage("Load a photo and wait for the model to finish loading.");
      return;
    }

    setBusy(true);
    setMessage("");

    try {
      const found = await detectMoney(model, imageRef.current);
      const nextCounts = emptyCounts();

      found.forEach(({ currencyId }) => {
        if (currencyId in nextCounts) nextCounts[currencyId]++;
      });

      setDetections(found);
      setCounts(nextCounts);

      const total = DENOMINATIONS.reduce(
        (sum, item) => sum + nextCounts[item.id] * item.value,
        0
      );

      setMessage(
        found.length
          ? `We found ${found.length} item(s). The total is: $${total.toFixed(2)} CAD.`
          : "Put something infront of the camera next time !!"
      );
    } catch (error) {
      console.error(error);
      setMessage(`Uh oh: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  function changeCount(id, amount) {
    setCounts((current) => ({
      ...current,
      [id]: Math.max(0, current[id] + amount),
    }));
  }

  function resetCounts() {
    setCounts(emptyCounts());
    setDetections([]);
    setMessage("Count reset.");
  }

  const total = DENOMINATIONS.reduce(
    (sum, item) => sum + counts[item.id] * item.value,
    0
  );

  //Buncha styling below, sorry for the mess Dawid.
  return (
    <>
      <style>{`
- { box-sizing: border-box; }
        body {
          margin: 0;
          min-height: 100vh;
          background: #151715;
          color: #eeeeeb;
          font-family: Arial, Helvetica, sans-serif;
        }
        button, input { font: inherit; }
        button {
          padding: 9px 13px;
          border: 1px solid #414641;
          border-radius: 4px;
          background: #292d29;
          color: #eeeeeb;
          cursor: pointer;
        }
        button:hover { background: #353a35; }
        button:disabled { opacity: .5; cursor: not-allowed; }
        button:focus-visible { outline: 2px solid #8ab99a; outline-offset: 2px; }
        .app { width: min(960px, calc(100% - 32px)); margin: 36px auto; }
        header { margin-bottom: 24px; }
        h1 { margin: 0 0 3px; font-family: Georgia, "Times New Roman", serif; font-size: 46px; line-height: 1.08; font-weight: 500; letter-spacing: -1px; }
        header p { margin: 0 0 8px; line-height: 1.4; color: #a5aaa5; }
        header small { color: #91b99a; }
        .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
        .controls input { display: none; }
        .controls button:last-child { background: #779b80; color: #101710; border-color: #779b80; }
        .controls button:last-child:hover { background: #89aa91; }
        .workspace { display: grid; grid-template-columns: 1.3fr 1fr; gap: 14px; }
        .preview, .results {
          min-width: 0;
          padding: 16px;
          background: #1d211d;
          border: 1px solid #343934;
          border-radius: 5px;
        }
        .preview {
          min-height: 350px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          overflow: hidden;
        }
        .image-wrap { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
        .preview img { display: block; max-width: 100%; max-height: 500px; object-fit: contain; }
        .detection-boxes { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
        .detection-box { fill: none; stroke: #80bd8d; stroke-width: 3; vector-effect: non-scaling-stroke; }
        .detection-label {
          fill: #b1d9b9;
          font-size: 22px;
          font-weight: 600;
          stroke: #151715;
          stroke-width: 4px;
          paint-order: stroke;
          stroke-linejoin: round;
        }
        .preview p, .hint { color: #a5aaa5; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; }
        .results { padding: 16px; }
        .total { padding-bottom: 16px; border-bottom: 1px solid #343934; }
        .total span { color: #a5aaa5; font-size: 13px; }
        .total strong { display: block; margin-top: 6px; color: #a8d2b0; font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums; }
        h2 { margin: 18px 0 8px; font-size: 14px; font-weight: 600; }
        .count-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #303530; font-size: 14px; }
        .count-row > div { display: flex; align-items: center; gap: 9px; }
        .count-row button { width: 28px; height: 28px; padding: 0; }
        .count-row strong { min-width: 18px; text-align: center; font-variant-numeric: tabular-nums; }
        .reset { margin-top: 16px; }
        .message { margin-top: 14px; padding: 10px 12px; color: #b9d7c0; background: #202820; border: 1px solid #3c5141; border-radius: 4px; font-size: 13px; line-height: 1.5; }
        @media (max-width: 680px) {
          .app { margin: 26px auto; }
          h1 { font-size: 38px; }
          .workspace { grid-template-columns: 1fr; }
          .preview { min-height: 260px; }
          .controls button { flex: 1 1 calc(50% - 8px); }
        }
      `}</style>
      <main className="app">
      <header>
        <h1>Canadian Money Counter</h1>
        <p>From the Looney Toonies for the DEV 2.0 Hackathon</p>
        <small>{modelStatus}</small>
      </header>

      <section className="controls">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={(event) => {
            choosePhoto(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <input
          ref={captureRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(event) => {
            choosePhoto(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <button onClick={() => fileRef.current?.click()}>Choose photo</button>
        <button onClick={openCamera}>Capture photo</button>
        <button onClick={countMoney} disabled={!model || !photo || busy}>
          {busy ? "Counting…" : "Count money"}
        </button>
      </section>

      {cameraOpen && (
        <div className="preview" style={{ marginBottom: 14 }}>
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ maxWidth: "100%", maxHeight: 400, borderRadius: 4 }}
          />
          <div className="controls" style={{ marginTop: 10, marginBottom: 0 }}>
            <button onClick={snapPhoto}>Snap photo</button>
            <button onClick={stopCamera}>Cancel</button>
          </div>
        </div>
      )}

      {message && <p className="message" role="status">{message}</p>}

      <section className="workspace">
        <div className="preview">
          {photo ? (
            <div className="image-wrap">
              <img
                ref={imageRef}
                src={photo}
                alt="Selected Canadian currency"
                onLoad={() => setMessage("")}
              />
              {detections.length > 0 && imageRef.current && (
                <svg
                  className="detection-boxes"
                  viewBox={`0 0 ${imageRef.current.naturalWidth} ${imageRef.current.naturalHeight}`}
                  preserveAspectRatio="none"
                  aria-label="Detected money marked with boxes"
                >
                  {detections.map((item, index) => (
                    <g key={`${item.currencyId}-${index}`}>
                      <rect
                        className="detection-box"
                        x={item.box.left}
                        y={item.box.top}
                        width={item.box.width}
                        height={item.box.height}
                      />
                      <text
                        className="detection-label"
                        x={item.box.left + 5}
                        y={Math.max(24, item.box.top - 7)}
                      >
                        {item.className}
                      </text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
          ) : (
            <p>Select or capture a photo to get started.</p>
          )}
          {fileName && <p>{fileName}</p>}
        </div>

        <div className="results">
          <div className="total">
            <span>Total</span>
            <strong>${total.toFixed(2)} CAD</strong>
          </div>

          {["Bill", "Coin"].map((type) => (
            <div key={type}>
              <h2>{type}s</h2>
              {DENOMINATIONS.filter((item) => item.type === type).map((item) => (
                <div className="count-row" key={item.id}>
                  <span>{item.name}</span>
                  <div>
                    <button onClick={() => changeCount(item.id, -1)} aria-label={`Remove ${item.name}`}>−</button>
                    <strong>{counts[item.id]}</strong>
                    <button onClick={() => changeCount(item.id, 1)} aria-label={`Add ${item.name}`}>+</button>
                  </div>
                </div>
              ))}
            </div>
          ))}

          <button className="reset" onClick={resetCounts}>Reset count</button>
        </div>
      </section>


      {detections.length > 0 && (
        <p className="hint">
          Detected items: {detections.map((item) => `${item.className} (${Math.round(item.score * 100)}%)`).join(", ")}
        </p>
      )}
      </main>
    </>
  );
}