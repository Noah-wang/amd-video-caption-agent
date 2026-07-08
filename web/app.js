const captions = {
  formal:
    "A busy urban street with two-way traffic and autumn trees under daytime lighting.",
  sarcastic:
    "Wow, cars and trees. Never seen that before. Totally groundbreaking.",
  humorous_tech:
    "Traffic API returns moderate latency; autumn trees deploy fall theme UI update.",
  humorous_non_tech:
    "Cars go vroom, trees go orange, and everyone pretends they aren't stuck.",
};

const labels = {
  formal: "formal",
  sarcastic: "sarcastic",
  humorous_tech: "humorous_tech",
  humorous_non_tech: "humorous_non_tech",
};

const tabs = document.querySelectorAll("[data-style]");
const captionText = document.querySelector("#captionText");
const styleLabel = document.querySelector("#styleLabel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");

    const style = tab.dataset.style;
    captionText.animate(
      [
        { opacity: 0, transform: "translateY(10px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      { duration: 260, easing: "ease-out" },
    );
    captionText.textContent = captions[style];
    styleLabel.textContent = labels[style];
  });
});

document.querySelector("#copyImage").addEventListener("click", async () => {
  const imageRef = document.querySelector("#imageRef").textContent;
  await navigator.clipboard.writeText(imageRef);
  const button = document.querySelector("#copyImage");
  button.textContent = "Copied";
  setTimeout(() => {
    button.textContent = "Copy image";
  }, 1200);
});

const demoForm = document.querySelector("#demoForm");
const videoUrlInput = document.querySelector("#videoUrl");
const demoStatus = document.querySelector("#demoStatus");
const generateButton = document.querySelector("#generateButton");
const contactPreview = document.querySelector("#contactPreview");
const generatedGrid = document.querySelector("#generatedGrid");
const factsBox = document.querySelector("#factsBox");

function setStatus(message) {
  demoStatus.textContent = message;
}

function waitForEvent(target, eventName) {
  return new Promise((resolve, reject) => {
    const onSuccess = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Video ${eventName} failed. The URL may be blocked or unsupported.`));
    };
    const cleanup = () => {
      target.removeEventListener(eventName, onSuccess);
      target.removeEventListener("error", onError);
    };

    target.addEventListener(eventName, onSuccess, { once: true });
    target.addEventListener("error", onError, { once: true });
  });
}

async function seekVideo(video, time) {
  video.currentTime = time;
  await waitForEvent(video, "seeked");
}

async function createContactSheet(videoUrl, frameCount = 4) {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = videoUrl;

  await waitForEvent(video, "loadedmetadata");

  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 8;
  const frameWidth = 384;
  const frameHeight = Math.round(frameWidth * 9 / 16);
  const canvas = document.createElement("canvas");
  const columns = 2;
  const rows = Math.ceil(frameCount / columns);
  canvas.width = columns * frameWidth;
  canvas.height = rows * frameHeight;
  const context = canvas.getContext("2d");

  context.fillStyle = "#06131d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.font = "18px IBM Plex Mono, monospace";

  for (let index = 0; index < frameCount; index += 1) {
    const timestamp = duration * (index + 1) / (frameCount + 1);
    await seekVideo(video, Math.min(timestamp, Math.max(duration - 0.1, 0)));

    const x = (index % columns) * frameWidth;
    const y = Math.floor(index / columns) * frameHeight;
    context.drawImage(video, x, y, frameWidth, frameHeight);
    context.fillStyle = "rgba(6, 19, 29, 0.72)";
    context.fillRect(x + 10, y + 10, 118, 32);
    context.fillStyle = "#f8edda";
    context.fillText(`t=${timestamp.toFixed(1)}s`, x + 20, y + 32);
  }

  try {
    return canvas.toDataURL("image/jpeg", 0.82);
  } catch {
    throw new Error(
      "The browser could load the video, but CORS blocked frame extraction. Try a CORS-enabled MP4 URL.",
    );
  }
}

function renderLiveResult(result) {
  document.querySelector("#liveFormal").textContent = result.captions.formal;
  document.querySelector("#liveSarcastic").textContent = result.captions.sarcastic;
  document.querySelector("#liveTech").textContent = result.captions.humorous_tech;
  document.querySelector("#liveNonTech").textContent = result.captions.humorous_non_tech;
  document.querySelector("#liveFacts").textContent = result.visual_facts || "No visual facts returned.";
  generatedGrid.hidden = false;
  factsBox.hidden = false;
}

demoForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const videoUrl = videoUrlInput.value.trim();

  generateButton.disabled = true;
  generatedGrid.hidden = true;
  factsBox.hidden = true;
  contactPreview.hidden = true;

  try {
    setStatus("Sampling video frames in the browser...");
    const contactSheet = await createContactSheet(videoUrl);
    contactPreview.src = contactSheet;
    contactPreview.hidden = false;

    setStatus("Calling the secure Vercel API route...");
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoUrl, contactSheet }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Generation failed.");
    }

    renderLiveResult(payload);
    setStatus(`Generated captions with ${payload.model}.`);
  } catch (error) {
    setStatus(error.message || "Something went wrong.");
  } finally {
    generateButton.disabled = false;
  }
});
