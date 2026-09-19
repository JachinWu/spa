const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusElem = document.getElementById("status");
const fpsElem = document.getElementById("fps");

// 模型設定 (若使用 320 匯出模型請改為 320)
const MODEL_SIZE = 320; 
const MODEL_PATH = "./yolov8n.onnx";
const CONF_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;

// COCO 80 類別標籤
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

// 用於影像前處理的離屏 Canvas
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = MODEL_SIZE;
offscreenCanvas.height = MODEL_SIZE;
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

function showError(msg) {
  statusElem.innerText = `錯誤: ${msg}`;
  statusElem.style.background = "rgba(230, 40, 40, 0.85)";
  console.error(msg);
}

async function setupCamera() {
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
        reject(err);
      }
    };
    video.onerror = (e) => reject(e);
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

/**
 * 影像前處理：Letterbox 等比縮放並轉換為 NCHW Tensor
 */
function preprocess(videoElement) {
  const vw = videoElement.videoWidth;
  const vh = videoElement.videoHeight;
  
  // 計算等比例縮放與 Padding (Letterbox)
  const scale = Math.min(MODEL_SIZE / vw, MODEL_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = (MODEL_SIZE - nw) / 2;
  const padY = (MODEL_SIZE - nh) / 2;

  // 繪製至離屏 Canvas
  offscreenCtx.fillStyle = "#808080";
  offscreenCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  offscreenCtx.drawImage(videoElement, 0, 0, vw, vh, padX, padY, nw, nh);

  const imgData = offscreenCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const pixels = imgData.data;

  // 轉換為 Float32Array [1, 3, MODEL_SIZE, MODEL_SIZE] (RGB 歸一化)
  const floatData = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const channelLength = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < channelLength; i++) {
    const r = pixels[i * 4] / 255.0;
    const g = pixels[i * 4 + 1] / 255.0;
    const b = pixels[i * 4 + 2] / 255.0;

    floatData[i] = r;                              // R 通道
    floatData[channelLength + i] = g;              // G 通道
    floatData[2 * channelLength + i] = b;          // B 通道
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  return { tensor, scale, padX, padY };
}

/**
 * 計算 IoU 供 NMS 使用
 */
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

/**
 * 後處理：解析輸出張量 [1, 84, 8400] 並執行 NMS
 */
function postprocess(outputTensor, scale, padX, padY) {
  const [_, channels, numBoxes] = outputTensor.dims;
  const data = outputTensor.data;
  const numClasses = channels - 4; // 84 - 4 = 80

  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    // 找出類別最大機率
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

      // 轉換回原圖座標 (扣除 padding 並除以 scale)
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

  // 根據分數降冪排序
  candidates.sort((a, b) => b.score - a.score);

  // NMS 過濾
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

/**
 * 將偵測結果精準繪製在螢幕上 (映射 CSS object-fit: cover 座標)
 */
function drawDetections(boxes) {
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  // object-fit: cover 映射縮放比例與平移偏移量
  const renderScale = Math.max(screenW / videoW, screenH / videoH);
  const offsetX = (screenW - videoW * renderScale) / 2;
  const offsetY = (screenH - videoH * renderScale) / 2;

  boxes.forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const label = COCO_CLASSES[item.classId] || `ID: ${item.classId}`;
    const score = Math.round(item.score * 100);

    // 映射到螢幕顯示像素
    const sx = x1 * renderScale + offsetX;
    const sy = y1 * renderScale + offsetY;
    const sw = (x2 - x1) * renderScale;
    const sh = (y2 - y1) * renderScale;

    // 若是車輛相關類別加重顯示
    const isVehicle = ["car", "motorcycle", "bus", "truck"].includes(label);
    const boxColor = isVehicle ? "#00ff88" : "#00bbff";

    // 繪製框線
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = isVehicle ? 3 : 2;
    ctx.strokeRect(sx, sy, sw, sh);

    // 繪製標籤背板與文字
    const text = `${label.toUpperCase()} ${score}%`;
    ctx.font = "bold 12px sans-serif";
    const textWidth = ctx.measureText(text).width;

    ctx.fillStyle = boxColor;
    ctx.fillRect(sx, sy - 20, textWidth + 10, 20);

    ctx.fillStyle = "#000";
    ctx.fillText(text, sx + 5, sy - 5);
  });
}

/**
 * 主推論迴圈
 */
async function runInference() {
  if (isProcessing || !session || video.readyState < 2) {
    requestAnimationFrame(runInference);
    return;
  }

  isProcessing = true;
  const startTime = performance.now();

  try {
    // 1. 前處理
    const { tensor, scale, padX, padY } = preprocess(video);

    // 2. 執行推論
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    // 3. 後處理
    const boxes = postprocess(outputTensor, scale, padX, padY);

    // 4. 繪製動態框
    drawDetections(boxes);

    // 計算 FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFrameTime = now;
      fpsElem.innerText = `FPS: ${fps}`;
    }
  } catch (err) {
    console.error("推論出錯:", err);
  } finally {
    isProcessing = false;
    requestAnimationFrame(runInference);
  }
}

async function init() {
  try {
    statusElem.innerText = "正在啟動鏡頭...";
    await setupCamera();
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    statusElem.innerText = "載入 YOLO 模型中 (約 10MB)...";

    // 設定 ONNX Runtime Web 使用 WebGL 進行硬體加速
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgl", "wasm"]
    });

    statusElem.innerText = "推論運作中";
    runInference();
  } catch (err) {
    showError(err.message || "載入失敗");
  }
}

init();
    // 改用 loadeddata，並設置超時防呆
    video.onloadeddata = async () => {
      try {
        await video.play();
        resolve(video);
      } catch (err) {
        reject(new Error("自動播放失敗，請點擊螢幕重試: " + err.message));
      }
    };

    video.onerror = (e) => {
      reject(new Error("Video 元素載入串流錯誤"));
    };

    // 5 秒逾時保護
    setTimeout(() => {
      if (video.readyState < 2) {
        reject(new Error("相機串流載入超時 (readyState: " + video.readyState + ")"));
      }
    }, 5000);
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

  ctx.scale(dpr, dpr);
}

function renderLoop() {
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    fpsElem.innerText = `FPS: ${fps}`;
  }

  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  const boxW = 220;
  const boxH = 75;
  const cx = window.innerWidth / 2 - boxW / 2;
  const cy = window.innerHeight / 2 - boxH / 2;

  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 3;
  ctx.strokeRect(cx, cy, boxW, boxH);

  ctx.fillStyle = "#00ff88";
  ctx.fillRect(cx, cy - 24, 120, 24);
  ctx.fillStyle = "#000";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText("PLATE TEST", cx + 8, cy - 7);

  requestAnimationFrame(renderLoop);
}

async function init() {
  try {
    statusElem.innerText = "正在偵測鏡頭設備...";
    await setupCamera();

    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    statusElem.innerText = "鏡頭就緒 (預覽中)";
    renderLoop();
  } catch (err) {
    showError(err.name ? `${err.name}: ${err.message}` : err.message);
  }
}

// 若有自動播放政策限制，允許點擊畫面喚醒播放
window.addEventListener("click", () => {
  if (video.srcObject && video.paused) {
    video.play().then(() => {
      statusElem.innerText = "鏡頭就緒 (預覽中)";
      renderLoop();
    }).catch(e => showError(e.message));
  }
}, { once: true });

init();
