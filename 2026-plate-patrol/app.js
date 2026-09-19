const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusElem = document.getElementById("status");
const fpsElem = document.getElementById("fps");
const timestampElem = document.getElementById("timestamp");
const guideBox = document.getElementById("guide-box");
const guideText = document.getElementById("guide-text");
const plateResultBadge = document.getElementById("plate-result-badge");
const shutterBtn = document.getElementById("shutter-btn");
const flashOverlay = document.getElementById("flash-overlay");

const MODEL_SIZE = 640;
const MODEL_PATH = "./plate_best.onnx";
const CONF_THRESHOLD = 0.35;   // 車牌專用模型，門檻略降以提升召回
const IOU_THRESHOLD = 0.45;

// === 車牌 OCR 模型 (fast-plate-ocr cct-s-v2-global) ===
const OCR_MODEL_PATH = "./plate_ocr.onnx";
const OCR_IMG_H = 64;          // 模型輸入高
const OCR_IMG_W = 128;         // 模型輸入寬
const OCR_SLOTS = 10;          // 最多 10 個字元槽
const OCR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"; // 37 類
const OCR_PAD_CHAR = "_";
let ocrSession = null;

// 車牌專用偵測模型僅有單一類別
const PLATE_CLASSES = ["license_plate"];

let session = null;
let isProcessing = false;
let isOcrRunning = false;
let lastOcrTime = 0;
let recognizedPlate = "";
let currentDetections = [];

// === 車牌偵測狀態機 ===
// SEARCHING: 尚未偵測到車牌
// LOCKED:    已偵測到車牌框，開始對車牌區域進行 OCR
// CONFIRMED: 已辨識出車牌字元
const STATE = { SEARCHING: "SEARCHING", LOCKED: "LOCKED", CONFIRMED: "CONFIRMED" };
let patrolState = STATE.SEARCHING;
// 車牌需連續數幀被偵測到才算「鎖定」，避免瞬間誤觸
let plateDetectedStreak = 0;
const LOCK_STREAK_REQUIRED = 3;      // 連續 3 幀偵測到車牌才鎖定
let framesWithoutPlate = 0;
const UNLOCK_GRACE_FRAMES = 15;      // 連續 15 幀無車牌才解除鎖定（避免抖動）
// 車牌辨識結果需多次一致才確認，降低誤判
let plateCandidateCounts = {};
// 目前最佳車牌偵測框（原始視訊像素座標 [x1,y1,x2,y2]），供 OCR 裁切使用
let bestPlateBox = null;

let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// 離屏 Canvas
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = MODEL_SIZE;
offscreenCanvas.height = MODEL_SIZE;
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

// OCR 裁切專用 Canvas
const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

function getFormattedDateTime() {
  const now = new Date();
  const Y = now.getFullYear();
  const M = String(now.getMonth() + 1).padStart(2, "0");
  const D = String(now.getDate()).padStart(2, "0");
  const h = String(now.getHours()).padStart(2, "0");
  const m = String(now.getMinutes()).padStart(2, "0");
  const s = String(now.getSeconds()).padStart(2, "0");
  return `${Y}-${M}-${D} ${h}:${m}:${s}`;
}

setInterval(() => {
  timestampElem.innerText = getFormattedDateTime();
}, 1000);
timestampElem.innerText = getFormattedDateTime();

function showError(msg) {
  statusElem.innerText = msg;
  statusElem.style.background = "rgba(230, 40, 40, 0.9)";
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
    video.onerror = () => reject(new Error("相機啟動異常"));
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

/**
 * 裁切「偵測到的車牌框」，送入 OCR ONNX 模型辨識字元。
 * 模型: fast-plate-ocr cct-s-v2-global
 *   輸入 [1,64,128,3] uint8 (RGB, NHWC)
 *   輸出 plate [1,10,37] → 每個字元槽對 37 類做 argmax
 */
async function recognizePlateText() {
  if (isOcrRunning || !ocrSession || video.readyState < 2) return;
  // 未鎖定車牌、或尚無車牌框時，不進行辨識
  if (patrolState === STATE.SEARCHING || !bestPlateBox) return;

  isOcrRunning = true;
  if (patrolState !== STATE.CONFIRMED) {
    guideText.innerText = "車牌辨識中…";
  }

  try {
    const videoW = video.videoWidth;
    const videoH = video.videoHeight;

    // bestPlateBox 為原始視訊像素座標 [x1,y1,x2,y2]，向外擴張避免切到字元
    const [bx1, by1, bx2, by2] = bestPlateBox;
    const padW = (bx2 - bx1) * 0.04;
    const padH = (by2 - by1) * 0.08;

    const cropX = Math.max(0, bx1 - padW);
    const cropY = Math.max(0, by1 - padH);
    const cropW = Math.min(videoW - cropX, (bx2 - bx1) + padW * 2);
    const cropH = Math.min(videoH - cropY, (by2 - by1) + padH * 2);

    if (cropW < 8 || cropH < 8) {
      guideText.innerText = "車牌已鎖定 - 對準車牌";
      return;
    }

    // 直接縮放到模型輸入尺寸 128x64（不保留長寬比，符合模型設定）
    cropCanvas.width = OCR_IMG_W;
    cropCanvas.height = OCR_IMG_H;
    cropCtx.drawImage(
      video,
      cropX, cropY, cropW, cropH,
      0, 0, OCR_IMG_W, OCR_IMG_H
    );

    const imgData = cropCtx.getImageData(0, 0, OCR_IMG_W, OCR_IMG_H);
    const d = imgData.data; // RGBA
    // 組成 uint8 NHWC RGB 張量 [1,64,128,3]
    const input = new Uint8Array(OCR_IMG_H * OCR_IMG_W * 3);
    for (let p = 0, o = 0; p < d.length; p += 4, o += 3) {
      input[o] = d[p];       // R
      input[o + 1] = d[p + 1]; // G
      input[o + 2] = d[p + 2]; // B
    }
    const tensor = new ort.Tensor("uint8", input, [1, OCR_IMG_H, OCR_IMG_W, 3]);

    const results = await ocrSession.run({ [ocrSession.inputNames[0]]: tensor });
    const plateOut = results["plate"] || results[ocrSession.outputNames[0]];
    const data = plateOut.data;      // [1,10,37]
    const numClasses = OCR_ALPHABET.length; // 37

    // 每個字元槽取 argmax
    let text = "";
    for (let slot = 0; slot < OCR_SLOTS; slot++) {
      let bestIdx = 0;
      let bestVal = -Infinity;
      const base = slot * numClasses;
      for (let c = 0; c < numClasses; c++) {
        const v = data[base + c];
        if (v > bestVal) { bestVal = v; bestIdx = c; }
      }
      const ch = OCR_ALPHABET[bestIdx];
      if (ch !== OCR_PAD_CHAR) text += ch;
    }

    const plate = normalizeTaiwanPlate(text) || (text.length >= 5 ? text : null);

    if (plate) {
      // 候選投票：取出現最多次者，穩定顯示並降低偶發誤判
      plateCandidateCounts[plate] = (plateCandidateCounts[plate] || 0) + 1;
      let best = plate;
      let bestCount = plateCandidateCounts[plate];
      for (const k in plateCandidateCounts) {
        if (plateCandidateCounts[k] > bestCount) { best = k; bestCount = plateCandidateCounts[k]; }
      }

      recognizedPlate = best;
      plateResultBadge.innerText = recognizedPlate;
      plateResultBadge.style.display = "block";

      if (bestCount >= 2 || normalizeTaiwanPlate(best)) {
        patrolState = STATE.CONFIRMED;
        guideText.innerText = "辨識成功：" + recognizedPlate;
      } else {
        guideText.innerText = "偵測到：" + recognizedPlate + "（比對確認中）";
      }
    } else {
      guideText.innerText = "車牌已鎖定 - 對準車牌";
    }
  } catch (err) {
    console.warn("OCR 失敗:", err);
    guideText.innerText = "辨識暫停，重試中…";
  } finally {
    isOcrRunning = false;
  }
}

/**
 * 驗證並正規化台灣車牌格式。
 * 支援常見格式：
 *   汽車新式  AAA-9999 / AAA9999 (3英+4數)
 *   汽車舊式  9999-AA  / 99AA / AA9999 等 (5~7碼英數混合)
 *   機車      AAA-999 / 999-AAA / AAA9999
 * 回傳格式化後字串（含連字號）或 null。
 */
function normalizeTaiwanPlate(s) {
  if (!s) return null;
  // 長度必須 5~7 碼英數字
  if (s.length < 5 || s.length > 7) return null;
  // 必須同時含有英文字母與數字（純字母或純數字通常是誤判）
  if (!/[A-Z]/.test(s) || !/[0-9]/.test(s)) return null;

  const patterns = [
    { re: /^([A-Z]{3})(\d{4})$/, fmt: (m) => `${m[1]}-${m[2]}` },   // AAA-9999
    { re: /^(\d{4})([A-Z]{2})$/, fmt: (m) => `${m[1]}-${m[2]}` },   // 9999-AA
    { re: /^([A-Z]{2})(\d{4})$/, fmt: (m) => `${m[1]}-${m[2]}` },   // AA-9999
    { re: /^([A-Z]{3})(\d{3})$/, fmt: (m) => `${m[1]}-${m[2]}` },   // AAA-999 (機車)
    { re: /^(\d{3})([A-Z]{3})$/, fmt: (m) => `${m[1]}-${m[2]}` },   // 999-AAA
    { re: /^(\d{2})([A-Z]{2})(\d{2})$/, fmt: (m) => `${m[1]}${m[2]}-${m[3]}` }, // 99AA-99
  ];
  for (const p of patterns) {
    const m = s.match(p.re);
    if (m) return p.fmt(m);
  }
  // 沒有完全符合格式，但長度合理且英數混合 → 視為未確認，回傳 null 讓其繼續嘗試
  return null;
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

  let plateFound = false;
  let bestScore = 0;
  let bestBoxVideo = null;

  boxes.forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const score = Math.round(item.score * 100);

    const sx = x1 * renderScale + offsetX;
    const sy = y1 * renderScale + offsetY;
    const sw = (x2 - x1) * renderScale;
    const sh = (y2 - y1) * renderScale;

    // 車牌框（綠色）
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 3;
    ctx.strokeRect(sx, sy, sw, sh);

    const text = `PLATE ${score}%`;
    ctx.font = "bold 13px sans-serif";
    const textWidth = ctx.measureText(text).width;
    ctx.fillStyle = "#00ff88";
    ctx.fillRect(sx, sy - 20, textWidth + 10, 20);
    ctx.fillStyle = "#000";
    ctx.fillText(text, sx + 5, sy - 5);

    plateFound = true;
    // 選出信心值最高的車牌作為 OCR 目標
    if (item.score > bestScore) {
      bestScore = item.score;
      bestBoxVideo = [x1, y1, x2, y2]; // 原始視訊像素座標
    }
  });

  if (plateFound) {
    framesWithoutPlate = 0;
    plateDetectedStreak++;
    bestPlateBox = bestBoxVideo;

    // 連續數幀偵測到車牌才正式鎖定
    if (patrolState === STATE.SEARCHING && plateDetectedStreak >= LOCK_STREAK_REQUIRED) {
      patrolState = STATE.LOCKED;
      plateCandidateCounts = {};
      guideText.innerText = "車牌已鎖定 - 辨識中…";
    }

    guideBox.classList.add("active");

    // 已鎖定（或已確認需持續更新）狀態才啟動 OCR
    if (patrolState === STATE.LOCKED || patrolState === STATE.CONFIRMED) {
      const now = performance.now();
      if (!isOcrRunning && now - lastOcrTime > 800) {
        lastOcrTime = now;
        recognizePlateText();
      }
    }
  } else {
    plateDetectedStreak = 0;
    framesWithoutPlate++;

    // 車牌離開需一段寬限幀數才解除鎖定，避免偵測抖動反覆重置
    if (framesWithoutPlate >= UNLOCK_GRACE_FRAMES) {
      if (patrolState !== STATE.SEARCHING) {
        patrolState = STATE.SEARCHING;
        plateCandidateCounts = {};
      }
      bestPlateBox = null;
      guideBox.classList.remove("active");
      guideText.innerText = "請將車牌對準此框";
    }
  }
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

    currentDetections = postprocess(outputTensor, scale, padX, padY);
    drawDetections(currentDetections);

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

/**
 * 拍照功能
 */
function takePhoto() {
  flashOverlay.style.opacity = "0.85";
  setTimeout(() => {
    flashOverlay.style.opacity = "0";
  }, 120);

  const captureCanvas = document.createElement("canvas");
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  captureCanvas.width = vw;
  captureCanvas.height = vh;
  const cCtx = captureCanvas.getContext("2d");

  // 1. 繪製視訊底圖
  cCtx.drawImage(video, 0, 0, vw, vh);

  // 2. 烙印當前辨識結果
  if (recognizedPlate) {
    cCtx.fillStyle = "#00ff88";
    cCtx.font = "bold 32px monospace";
    cCtx.fillText(`PLATE: ${recognizedPlate}`, 30, 60);
  }

  // 3. 右下角烙印時間
  const timeStr = getFormattedDateTime();
  cCtx.font = "bold 24px monospace";
  const tw = cCtx.measureText(timeStr).width;
  cCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
  cCtx.fillRect(vw - tw - 30, vh - 55, tw + 20, 36);
  cCtx.fillStyle = "#00ff88";
  cCtx.fillText(timeStr, vw - tw - 20, vh - 28);

  // 4. 儲存圖檔（跨平台：桌機用 download，行動裝置以 Blob URL 開啟供長按儲存）
  const dateTag = timeStr.replace(/[- :]/g, "");
  const fileName = `plate_${recognizedPlate || "scan"}_${dateTag}.jpg`;

  captureCanvas.toBlob((blob) => {
    if (!blob) {
      // 極端後備：改用 dataURL 直接下載
      const link = document.createElement("a");
      link.download = fileName;
      link.href = captureCanvas.toDataURL("image/jpeg", 0.92);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    }

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = fileName;
    link.href = url;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // 行動裝置（尤其 iOS Safari）常忽略 download 屬性，另開分頁供使用者長按儲存
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (isMobile) {
      window.open(url, "_blank");
    }

    // 釋放記憶體
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }, "image/jpeg", 0.92);
}

shutterBtn.addEventListener("click", takePhoto);

async function init() {
  try {
    statusElem.innerText = "1/3 啟動相機...";
    await setupCamera();
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    statusElem.innerText = "2/3 載入車牌偵測模型...";
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgl", "wasm"]
    });

    statusElem.innerText = "3/3 載入車牌 OCR 模型...";
    // OCR 模型輸入為 uint8，WebGL 不支援整數張量，改用 wasm
    ocrSession = await ort.InferenceSession.create(OCR_MODEL_PATH, {
      executionProviders: ["wasm"]
    });

    statusElem.innerText = "巡邏中";
    runInference();
  } catch (err) {
    showError(`錯誤: ${err.message || err}`);
  }
}


init();
