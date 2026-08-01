const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("prediction-form");
const loader = document.getElementById("loader");

const probabilityText = document.getElementById("probability");
const predictionText = document.getElementById("prediction");
const riskText = document.getElementById("risk");
const thresholdText = document.getElementById("threshold");
const versionText = document.getElementById("version");
const modelText = document.getElementById("model");

const progressCircle = document.getElementById("progress");

const apiStatus = document.getElementById("api-status");
const statusDot = document.getElementById("status-dot");

const shapList = document.getElementById("shap-list");

const toast = document.getElementById("toast");

const radius = 75;
const circumference = 2 * Math.PI * radius;

progressCircle.style.strokeDasharray = circumference;
progressCircle.style.strokeDashoffset = circumference;

// ── Helpers ──
function showLoader() {
  loader.classList.remove("hidden");
}

function hideLoader() {
  loader.classList.add("hidden");
}

function showToast(message, color = "#22c55e") {
  toast.innerHTML = message;
  toast.style.borderLeftColor = color;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3000);
}

// ── API Health ──
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    apiStatus.innerHTML = "API Online";
    statusDot.style.background = "#22c55e";
  } catch {
    apiStatus.innerHTML = "API Offline";
    statusDot.style.background = "#ef4444";
  }
}

// ── Gauge ──
function setGauge(probability) {
  const percent = Math.min(Math.max(probability * 100, 0), 100);
  const offset = circumference - (percent / 100) * circumference;
  progressCircle.style.strokeDashoffset = offset;
  if (percent < 30) {
    progressCircle.style.stroke = "#22c55e";
  } else if (percent < 60) {
    progressCircle.style.stroke = "#f59e0b";
  } else {
    progressCircle.style.stroke = "#ef4444";
  }
}

// ── SHAP Rendering ──
function renderShap(features) {
  shapList.innerHTML = "";
  if (!features || features.length === 0) {
    shapList.innerHTML =
      '<p class="shap-placeholder">No explanation available.</p>';
    return;
  }
  features.forEach((item) => {
    const div = document.createElement("div");
    div.className = "shap-item";
    div.innerHTML = `
      <span class="shap-feature">${item.feature}</span>
      <span class="shap-impact">${Number(item.shap_value).toFixed(4)}</span>
    `;
    shapList.appendChild(div);
  });
}

// ── Form Validation ──
function validate() {
  const age = Number(document.getElementById("age").value);
  const bmi = Number(document.getElementById("bmi").value);
  const glucose = Number(document.getElementById("avg_glucose_level").value);

  if (isNaN(age) || age < 0 || age > 120) {
    showToast("Age must be between 0 and 120.", "#ef4444");
    return false;
  }
  if (isNaN(bmi) || bmi <= 0) {
    showToast("BMI must be a positive number.", "#ef4444");
    return false;
  }
  if (isNaN(glucose) || glucose <= 0) {
    showToast("Glucose level must be positive.", "#ef4444");
    return false;
  }
  return true;
}

// ── Build Payload ──
function payload() {
  return {
    gender: document.getElementById("gender").value,
    age: Number(document.getElementById("age").value),
    hypertension: Number(document.getElementById("hypertension").value),
    heart_disease: Number(document.getElementById("heart_disease").value),
    ever_married: document.getElementById("ever_married").value,
    work_type: document.getElementById("work_type").value,
    Residence_type: document.getElementById("Residence_type").value,
    avg_glucose_level: Number(
      document.getElementById("avg_glucose_level").value,
    ),
    bmi: Number(document.getElementById("bmi").value),
    smoking_status: document.getElementById("smoking_status").value,
  };
}

// ── Predict ──
async function predict(event) {
  event.preventDefault();
  if (!validate()) return;

  showLoader();
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    if (!response.ok) throw new Error("Prediction failed");

    const data = await response.json();

    probabilityText.innerHTML = (data.probability * 100).toFixed(2) + "%";
    predictionText.innerHTML = data.prediction;
    riskText.innerHTML = data.risk;
    thresholdText.innerHTML = data.threshold;
    versionText.innerHTML = data.version;
    modelText.innerHTML = data.model;

    setGauge(data.probability);

    // Assign risk class for colouring
    riskText.className = data.risk.toLowerCase();

    renderShap(data.top_features);

    showToast("Prediction completed successfully.");
  } catch (error) {
    showToast("Prediction failed. Please try again.", "#ef4444");
  } finally {
    hideLoader();
  }
}

// ── Preset Loaders (for demo) ──
function loadHighRisk() {
  document.getElementById("gender").value = "Male";
  document.getElementById("age").value = 67;
  document.getElementById("hypertension").value = 1;
  document.getElementById("heart_disease").value = 1;
  document.getElementById("ever_married").value = "Yes";
  document.getElementById("work_type").value = "Private";
  document.getElementById("Residence_type").value = "Urban";
  document.getElementById("avg_glucose_level").value = 228.69;
  document.getElementById("bmi").value = 36.6;
  document.getElementById("smoking_status").value = "formerly smoked";
}

function loadLowRisk() {
  document.getElementById("gender").value = "Female";
  document.getElementById("age").value = 24;
  document.getElementById("hypertension").value = 0;
  document.getElementById("heart_disease").value = 0;
  document.getElementById("ever_married").value = "No";
  document.getElementById("work_type").value = "Private";
  document.getElementById("Residence_type").value = "Urban";
  document.getElementById("avg_glucose_level").value = 90;
  document.getElementById("bmi").value = 22;
  document.getElementById("smoking_status").value = "never smoked";
}

// ── Event Listeners ──
form.addEventListener("submit", predict);

// Health check on load and every 10s
checkHealth();
setInterval(checkHealth, 10000);

// Load a high‑risk example on startup
loadHighRisk();
