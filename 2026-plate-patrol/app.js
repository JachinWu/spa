const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusElem = document.getElementById("status");
const fpsElem = document.getElementById("fps");

let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

function showError(msg) {
  statusElem.innerText = `錯誤: ${msg}`;
  statusElem.style.background = "rgba(230, 40, 40, 0.85)";
  console.error(msg);
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("此瀏覽器環境不支援或未允許 getUserMedia (請確認使用 HTTPS)");
  }

  // 使用寬鬆約束，避免部分手機硬體直接拋出 OverconstrainedError
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  statusElem.innerText = "等待鏡頭權限允許...";
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  return new Promise((resolve, reject) => {
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
