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
const loadingPanel = document.getElementById("loadingPanel");

const catProbText = document.getElementById("catProbText");
const dogProbText = document.getElementById("dogProbText");
const catBar = document.getElementById("catBar");
const dogBar = document.getElementById("dogBar");

const errorWrap = document.getElementById("errorWrap");

const themeToggle = document.getElementById("themeToggle");

let selectedFile = null;

const modelLoaded = window.__MODEL_LOADED__ === true;
const modelError = window.__MODEL_ERROR__ || "";

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function initTheme() {
  if (!themeToggle) return;

  try {
    const saved = window.localStorage.getItem("theme");
    const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
    const shouldDark = saved ? saved === "dark" : !!prefersDark;

    document.documentElement.classList.toggle("dark", shouldDark);
    themeToggle.textContent = shouldDark ? "Light mode" : "Dark mode";
  } catch (_e) {
    // If localStorage is blocked, just leave the default theme.
  }
}

function toggleTheme() {
  try {
    const nowDark = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", nowDark);
    window.localStorage.setItem("theme", nowDark ? "dark" : "light");
    if (themeToggle) themeToggle.textContent = nowDark ? "Light mode" : "Dark mode";
  } catch (_e) {
    // Ignore if theme persistence isn't available.
  }
}

initTheme();
if (themeToggle) {
  themeToggle.addEventListener("click", toggleTheme);
}

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

  if (catProbText) catProbText.textContent = "—";
  if (dogProbText) dogProbText.textContent = "—";
  if (catBar) catBar.style.width = "0%";
  if (dogBar) dogBar.style.width = "0%";

  if (loadingPanel) loadingPanel.classList.add("hidden");
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading || !selectedFile || !modelLoaded;
  spinner.classList.toggle("hidden", !isLoading);
  if (loadingPanel) loadingPanel.classList.toggle("hidden", !isLoading);
  predictBtnText.textContent = isLoading ? "Predicting" : "Predict";
  statusText.textContent = isLoading
    ? "Running model inference…"
    : selectedFile
      ? "Click Predict to continue."
      : modelLoaded
        ? "Select an image to enable prediction."
        : modelError
          ? `Model not loaded: ${modelError}`
          : "Model not loaded. Check server logs and restart.";
}

function setFile(file) {
  clearError();
  resetResult();

  selectedFile = file || null;
  if (!selectedFile) {
    previewWrap.classList.add("hidden");
    clearBtn.classList.add("hidden");
    predictBtn.disabled = true;
    statusText.textContent = modelLoaded
      ? "Select an image to enable prediction."
      : modelError
        ? `Model not loaded: ${modelError}`
        : "Model not loaded. Check server logs and restart.";
    fileInput.value = "";
    return;
  }

  clearBtn.classList.remove("hidden");
  previewWrap.classList.remove("hidden");

  fileMeta.textContent = `${selectedFile.name} • ${humanFileSize(selectedFile.size)}`;

  const url = URL.createObjectURL(selectedFile);
  previewImg.onload = () => URL.revokeObjectURL(url);
  previewImg.src = url;

  predictBtn.disabled = !modelLoaded;
  statusText.textContent = modelLoaded
    ? "Click Predict to continue."
    : modelError
      ? `Model not loaded: ${modelError}`
      : "Model not loaded. Check server logs and restart.";
}

function isValidImageFile(file) {
  if (!file) return false;
  const okTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp"];
  if (okTypes.includes(file.type)) return true;
  const name = (file.name || "").toLowerCase();
  return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".bmp");
}

function isValidSize(file) {
  if (!file) return false;
  return file.size <= MAX_FILE_BYTES;
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

  if (file && !isValidSize(file)) {
    setFile(null);
    showError("File too large. Please choose an image up to 10MB.");
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

  if (file && !isValidSize(file)) {
    setFile(null);
    showError("File too large. Please drop an image up to 10MB.");
    return;
  }

  setFile(file || null);
});

function renderResult(data) {
  if (loadingPanel) loadingPanel.classList.add("hidden");

  const label = data.label;
  const confidence = Number(data.confidence);
  const rawScore = Number(data.raw_score);

  const lowerLabel = String(label).toLowerCase();
  let badgeClass = "bg-slate-100 text-slate-700 ring-1 ring-slate-200";
  
  if (lowerLabel === "cat") {
    badgeClass = "bg-violet-50 text-violet-800 ring-1 ring-violet-200";
  } else if (lowerLabel === "dog") {
    badgeClass = "bg-sky-50 text-sky-800 ring-1 ring-sky-200";
  }

  resultBadge.className = `rounded-full px-4 py-2 text-sm font-semibold ${badgeClass}`;
  resultBadge.textContent = label;

  const confidencePct = Number.isFinite(confidence) ? clamp01(confidence) * 100 : 0;
  confidenceText.textContent = `${confidencePct.toFixed(1)}%`;
  rawScoreText.textContent = Number.isFinite(rawScore) ? rawScore.toFixed(4) : "—";

  const dogProb = Number.isFinite(data.dog_prob) ? clamp01(Number(data.dog_prob)) : clamp01(rawScore);
  const catProb = Number.isFinite(data.cat_prob) ? clamp01(Number(data.cat_prob)) : 1.0 - dogProb;

  if (catProbText) catProbText.textContent = `${(catProb * 100).toFixed(1)}%`;
  if (dogProbText) dogProbText.textContent = `${(dogProb * 100).toFixed(1)}%`;
  if (catBar) catBar.style.width = `${(catProb * 100).toFixed(0)}%`;
  if (dogBar) dogBar.style.width = `${(dogProb * 100).toFixed(0)}%`;

  resultWrap.classList.remove("hidden");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  if (!modelLoaded) {
    showError(modelError || "Model not loaded on the server.");
    return;
  }
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

// Set initial state based on server model availability.
if (!modelLoaded) {
  predictBtn.disabled = true;
  statusText.textContent = modelError
    ? `Model not loaded: ${modelError}`
    : "Model not loaded. Check server logs and restart.";
}

