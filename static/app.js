const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const fileMeta = document.getElementById("fileMeta");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictBtnText = document.getElementById("predictBtnText");
const spinner = document.getElementById("spinner");
const statusText = document.getElementById("statusText");

const resultWrap = document.getElementById("resultWrap");
const resultBadge = document.getElementById("resultBadge");
const confidenceText = document.getElementById("confidenceText");
const rawScoreText = document.getElementById("rawScoreText");
const confidenceBar = document.getElementById("confidenceBar");

const errorWrap = document.getElementById("errorWrap");

let selectedFile = null;

function humanFileSize(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let num = bytes;
  while (num >= 1024 && i < units.length - 1) {
    num /= 1024;
    i += 1;
  }
  return `${num.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function setDragging(on) {
  dropzone.classList.toggle("is-dragging", on);
}

function showError(message) {
  errorWrap.textContent = message;
  errorWrap.classList.remove("hidden");
}

function clearError() {
  errorWrap.textContent = "";
  errorWrap.classList.add("hidden");
}

function resetResult() {
  resultWrap.classList.add("hidden");
  resultBadge.textContent = "";
  confidenceText.textContent = "";
  rawScoreText.textContent = "";
  confidenceBar.style.width = "0%";
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading || !selectedFile;
  spinner.classList.toggle("hidden", !isLoading);
  predictBtnText.textContent = isLoading ? "Predicting" : "Predict";
  statusText.textContent = isLoading
    ? "Running model inference…"
    : selectedFile
      ? "Click Predict to continue."
      : "Select an image to enable prediction.";
}

function setFile(file) {
  clearError();
  resetResult();

  selectedFile = file || null;
  if (!selectedFile) {
    previewWrap.classList.add("hidden");
    clearBtn.classList.add("hidden");
    predictBtn.disabled = true;
    statusText.textContent = "Select an image to enable prediction.";
    fileInput.value = "";
    return;
  }

  clearBtn.classList.remove("hidden");
  previewWrap.classList.remove("hidden");

  fileMeta.textContent = `${selectedFile.name} • ${humanFileSize(selectedFile.size)}`;

  const url = URL.createObjectURL(selectedFile);
  previewImg.onload = () => URL.revokeObjectURL(url);
  previewImg.src = url;

  predictBtn.disabled = false;
  statusText.textContent = "Click Predict to continue.";
}

function isValidImageFile(file) {
  if (!file) return false;
  const okTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp"];
  if (okTypes.includes(file.type)) return true;
  const name = (file.name || "").toLowerCase();
  return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".bmp");
}

dropzone.addEventListener("click", (e) => {
  const isButton = e.target?.closest?.("button, a, label");
  if (isButton) return;
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file && !isValidImageFile(file)) {
    setFile(null);
    showError("Please select a valid image file (PNG, JPG, JPEG, BMP).");
    return;
  }
  setFile(file);
});

clearBtn.addEventListener("click", () => setFile(null));

dropzone.addEventListener("dragenter", (e) => {
  e.preventDefault();
  setDragging(true);
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  setDragging(true);
});
dropzone.addEventListener("dragleave", (e) => {
  if (e.target === dropzone) setDragging(false);
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  setDragging(false);
  const file = e.dataTransfer?.files?.[0];
  if (file && !isValidImageFile(file)) {
    setFile(null);
    showError("Please drop a valid image file (PNG, JPG, JPEG, BMP).");
    return;
  }
  setFile(file || null);
});

function renderResult(data) {
  const label = data.label;
  const confidence = Number(data.confidence);
  const rawScore = Number(data.raw_score);

  const isCat = String(label).toLowerCase().includes("cat");

  const badgeClass = isCat
    ? "bg-violet-50 text-violet-800 ring-1 ring-violet-200"
    : "bg-sky-50 text-sky-800 ring-1 ring-sky-200";

  resultBadge.className = `rounded-full px-4 py-2 text-sm font-semibold ${badgeClass}`;
  resultBadge.textContent = label;

  const pct = Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) * 100 : 0;
  confidenceText.textContent = `${pct.toFixed(1)}%`;
  rawScoreText.textContent = Number.isFinite(rawScore) ? rawScore.toFixed(4) : "—";
  confidenceBar.style.width = `${pct.toFixed(0)}%`;

  resultWrap.classList.remove("hidden");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  clearError();
  resetResult();
  setLoading(true);

  try {
    const fd = new FormData();
    fd.append("image", selectedFile);

    const res = await fetch(window.__PREDICT_URL__ || "/predict", {
      method: "POST",
      body: fd,
    });

    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const message = data?.error || `Request failed (${res.status}).`;
      throw new Error(message);
    }
    if (!data || !data.label) {
      throw new Error("Unexpected response from server.");
    }
    renderResult(data);
  } catch (err) {
    showError(err?.message || "Something went wrong while predicting.");
  } finally {
    setLoading(false);
  }
});

