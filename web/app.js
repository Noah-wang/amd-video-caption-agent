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
