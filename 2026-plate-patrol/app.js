const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusElem = document.getElementById("status");
const fpsElem = document.getElementById("fps");

// 設定：因為匯出時是 imgsz=320，此處鎖定 320
const MODEL_SIZE = 320;
const MODEL_PATH = "./yolov8n.onnx";
const CONF_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;

const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

let session = null;
let isProcessing = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// 全域錯誤捕捉，直接顯示在螢幕上
window.addEventListener("error", (e) => {
  showError(`全域錯誤: ${e.message} (行號: ${e.lineno})`);
});

function showError(msg) {
  statusElem.innerText = msg;
  statusElem.style.background = "rgba(230, 40, 40, 0.9)";
  console.error(msg);
}

// 離屏 Canvas 供前處理抽幀
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = MODEL_SIZE;
offscreenCanvas.height = MODEL_SIZE;
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("瀏覽器不支援相機或非 HTTPS 環境");
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  return new Promise((resolve, reject) => {
    video.onloadeddata = async () => {
      try {
        await video.play();
        resolve(video);
      } catch (err) {
        reject(new Error("自動播放被阻擋，請點擊螢幕: " + err.message));
      }
    };
    video.onerror = () => reject(new Error("Video 元素載入串流失敗"));
  });
}

function updateCanvasSize() {
  const dpr = window.devicePixelRatio || 1;
  const width = window.innerWidth;
  const height = window.innerHeight;

  overlay.width = width * dpr;
  overlay.height = height * dpr;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

function preprocess(videoElement) {
  const vw = videoElement.videoWidth;
  const vh = videoElement.videoHeight;

  const scale = Math.min(MODEL_SIZE / vw, MODEL_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = (MODEL_SIZE - nw) / 2;
  const padY = (MODEL_SIZE - nh) / 2;

  offscreenCtx.fillStyle = "#808080";
  offscreenCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  offscreenCtx.drawImage(videoElement, 0, 0, vw, vh, padX, padY, nw, nh);

  const imgData = offscreenCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const pixels = imgData.data;

  const floatData = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const channelLength = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < channelLength; i++) {
    floatData[i] = pixels[i * 4] / 255.0;
    floatData[channelLength + i] = pixels[i * 4 + 1] / 255.0;
    floatData[2 * channelLength + i] = pixels[i * 4 + 2] / 255.0;
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  return { tensor, scale, padX, padY };
}

function iou(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  return interArea / (areaA + areaB - interArea);
}

function postprocess(outputTensor, scale, padX, padY) {
  // 安全取出維度，避免解構語法問題
  const channels = outputTensor.dims[1];
  const numBoxes = outputTensor.dims[2];
  const data = outputTensor.data;
  const numClasses = channels - 4;

  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    let maxScore = 0;
    let classId = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore >= CONF_THRESHOLD) {
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

      const x1 = (cx - w / 2 - padX) / scale;
      const y1 = (cy - h / 2 - padY) / scale;
      const x2 = (cx + w / 2 - padX) / scale;
      const y2 = (cy + h / 2 - padY) / scale;

      candidates.push({
        box: [x1, y1, x2, y2],
        score: maxScore,
        classId: classId
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  const finalBoxes = [];
  while (candidates.length > 0) {
    const current = candidates.shift();
    finalBoxes.push(current);

    for (let j = candidates.length - 1; j >= 0; j--) {
      if (current.classId === candidates[j].classId && iou(current.box, candidates[j].box) > IOU_THRESHOLD) {
        candidates.splice(j, 1);
      }
    }
  }

  return finalBoxes;
}

function drawDetections(boxes) {
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  const renderScale = Math.max(screenW / videoW, screenH / videoH);
  const offsetX = (screenW - videoW * renderScale) / 2;
  const offsetY = (screenH - videoH * renderScale) / 2;

  boxes.forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const label = COCO_CLASSES[item.classId] || `ID: ${item.classId}`;
    const score = Math.round(item.score * 100);

    const sx = x1 * renderScale + offsetX;
    const sy = y1 * renderScale + offsetY;
    const sw = (x2 - x1) * renderScale;
    const sh = (y2 - y1) * renderScale;

    const isVehicle = ["car", "motorcycle", "bus", "truck"].includes(label);
    const boxColor = isVehicle ? "#00ff88" : "#00bbff";

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = isVehicle ? 3 : 2;
    ctx.strokeRect(sx, sy, sw, sh);

    const text = `${label.toUpperCase()} ${score}%`;
    ctx.font = "bold 12px sans-serif";
    const textWidth = ctx.measureText(text).width;

    ctx.fillStyle = boxColor;
    ctx.fillRect(sx, sy - 20, textWidth + 10, 20);

    ctx.fillStyle = "#000";
    ctx.fillText(text, sx + 5, sy - 5);
  });
}

async function runInference() {
  if (isProcessing || !session || video.readyState < 2) {
    requestAnimationFrame(runInference);
    return;
  }

  isProcessing = true;

  try {
    const { tensor, scale, padX, padY } = preprocess(video);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    const boxes = postprocess(outputTensor, scale, padX, padY);
    drawDetections(boxes);

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFrameTime = now;
      fpsElem.innerText = `FPS: ${fps}`;
    }
  } catch (err) {
    console.error("推論錯誤:", err);
  } finally {
    isProcessing = false;
    requestAnimationFrame(runInference);
  }
}

async function init() {
  try {
    statusElem.innerText = "1/3 正在啟動鏡頭...";
    await setupCamera();
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    statusElem.innerText = "2/3 載入 YOLO 模型中 (約 6MB)...";
    
    // 檢查 onnxruntime 是否正確載入
    if (typeof ort === "undefined") {
      throw new Error("ort.min.js CDN 未能成功載入，請檢查網路連線");
    }

    ort.env.wasm.numThreads = 1;
    // 優先 WebGL，若瀏覽器不支援則降級至 WASM
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgl", "wasm"]
    });

    statusElem.innerText = "3/3 偵測運作中";
    runInference();
  } catch (err) {
    showError(`錯誤: ${err.message || err}`);
  }
}

// 點擊喚醒播放保險機制
window.addEventListener("click", () => {
  if (video.srcObject && video.paused) {
    video.play().catch(e => showError("播放失敗: " + e.message));
  }
}, { once: true });

init();
