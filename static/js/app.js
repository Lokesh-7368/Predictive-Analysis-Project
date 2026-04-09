const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const browseButton = document.getElementById("browse-button");
const predictButton = document.getElementById("predict-button");
const trainButton = document.getElementById("train-button");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const trainingConsole = document.getElementById("training-console");
const consoleStatus = document.getElementById("console-status");
const emptyState = document.getElementById("empty-state");
const resultState = document.getElementById("result-state");
const resultLabel = document.getElementById("result-label");
const confidenceRing = document.getElementById("confidence-ring");
const confidenceValue = document.getElementById("confidence-value");
const resultDescription = document.getElementById("result-description");
const probabilityList = document.getElementById("probability-list");
const trainingEnabled = trainButton?.dataset.trainingEnabled !== "false";

const state = {
    file: null,
    base64Image: "",
};

function formatFileSize(bytes) {
    if (!bytes) {
        return "0 KB";
    }

    const kb = bytes / 1024;
    if (kb < 1024) {
        return `${kb.toFixed(1)} KB`;
    }

    return `${(kb / 1024).toFixed(2)} MB`;
}

function setConsole(message, tone = "idle") {
    trainingConsole.textContent = message;
    consoleStatus.textContent = tone.charAt(0).toUpperCase() + tone.slice(1);
    consoleStatus.className = `console-status ${tone}`;
}

function clearResult() {
    emptyState.hidden = false;
    resultState.hidden = true;
    probabilityList.innerHTML = "";
}

function renderProbabilities(rows) {
    probabilityList.innerHTML = "";

    rows.forEach((row) => {
        const item = document.createElement("div");
        item.className = "probability-item";

        item.innerHTML = `
            <div class="probability-meta">
                <span>${row.label}</span>
                <strong>${row.probability.toFixed(2)}%</strong>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: ${row.probability}%;"></div>
            </div>
        `;

        probabilityList.appendChild(item);
    });
}

function renderResult(result) {
    emptyState.hidden = true;
    resultState.hidden = false;
    resultLabel.textContent = result.display_label;
    confidenceValue.textContent = `${result.confidence.toFixed(2)}%`;
    confidenceRing.style.setProperty("--confidence", `${result.confidence}%`);
    resultDescription.textContent = result.description;
    renderProbabilities(result.probabilities);
}

function readFile(file) {
    const reader = new FileReader();

    reader.onload = (event) => {
        const dataUrl = event.target.result;
        state.base64Image = dataUrl.split(",")[1];
        previewImage.src = dataUrl;
        previewImage.style.display = "block";
        previewPlaceholder.style.display = "none";
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        predictButton.disabled = false;
        clearResult();
    };

    reader.readAsDataURL(file);
}

function handleFile(file) {
    if (!file) {
        return;
    }

    state.file = file;
    readFile(file);
}

browseButton.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (event) => {
    handleFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    handleFile(file);
});

predictButton.addEventListener("click", async () => {
    if (!state.base64Image) {
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: state.base64Image }),
        });

        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.message || "Prediction request failed.");
        }

        renderResult(payload.result);
    } catch (error) {
        clearResult();
        alert(error.message);
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Analyze CT Scan";
    }
});

trainButton.addEventListener("click", async () => {
    if (!trainingEnabled) {
        setConsole(
            "This deployment is configured for prediction only. Use your local Python environment to run the DVC pipeline.",
            "idle"
        );
        return;
    }

    trainButton.disabled = true;
    trainButton.textContent = "Running Pipeline...";
    setConsole("Starting full pipeline...", "running");

    try {
        const response = await fetch("/train", {
            method: "POST",
        });

        const payload = await response.json();
        const tone = payload.success ? "success" : "error";
        const message = payload.log || payload.message || "No log output was returned.";
        setConsole(message, tone);

        if (!response.ok || !payload.success) {
            throw new Error(payload.message || "Training pipeline failed.");
        }
    } catch (error) {
        setConsole(error.message, "error");
        alert(error.message);
    } finally {
        trainButton.disabled = false;
        trainButton.textContent = "Run Full Pipeline";
    }
});
