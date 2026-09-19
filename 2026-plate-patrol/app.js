const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusElem = document.getElementById("status");
const fpsElem = document.getElementById("fps");

// FPS 計算用
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

/**
 * 啟動相機串流
 */
async function setupCamera() {
  // 檢查瀏覽器是否支援
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("此瀏覽器不支援相機存取功能");
  }

  // 設定參數：優先後置鏡頭、720p（手機推論最兼顧流暢度的解析度）
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve(video);
    };
  });
}

/**
 * 根據視窗尺寸調整 Overlay Canvas 解析度，避免繪圖模糊與拉伸
 */
function updateCanvasSize() {
  const dpr = window.devicePixelRatio || 1;
  const width = window.innerWidth;
  const height = window.innerHeight;

  // 實際渲染像素
  overlay.width = width * dpr;
  overlay.height = height * dpr;

  // CSS 顯示大小
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;

  // 縮放繪圖上下文以配合 DPR
  ctx.scale(dpr, dpr);
}

/**
 * 主渲染迴圈（Step 1 先做繪圖測試與 FPS 統計）
 */
function renderLoop() {
  // 1. 計算 FPS
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    fpsElem.innerText = `FPS: ${fps}`;
  }

  // 2. 清空前一幀畫布
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  // 3. 測試繪圖：在畫面正中央畫一個脈衝測試框（代表未來 YOLO 預測框的位置）
  const boxW = 200;
  const boxH = 70;
  const cx = window.innerWidth / 2 - boxW / 2;
  const cy = window.innerHeight / 2 - boxH / 2;

  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.strokeRect(cx, cy, boxW, boxH);

  // 繪製模擬標籤
  ctx.fillStyle = "#00ff88";
  ctx.fillRect(cx, cy - 22, 110, 22);
  ctx.fillStyle = "#000";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("PLATE 98%", cx + 6, cy - 6);

  // 持續下一個畫面幀
  requestAnimationFrame(renderLoop);
}

/**
 * 初始化入口
 */
async function init() {
  try {
    statusElem.innerText = "請求相機權限中...";
    await setupCamera();
    
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    statusElem.innerText = "鏡頭就緒 (純預覽)";
    renderLoop();
  } catch (err) {
    console.error(err);
    statusElem.innerText = `錯誤: ${err.message || "無法存取鏡頭"}`;
    statusElem.style.background = "rgba(255, 0, 0, 0.6)";
  }
}

// 啟動
init();
