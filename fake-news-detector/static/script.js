/**
 * TruthLens – Fake News Detector
 * Frontend Interactions Script
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const newsInput = document.getElementById("newsInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const clearBtn = document.getElementById("clearBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnIcon = document.getElementById("btnIcon");
    const btnText = document.getElementById("btnText");
    const wordCountSpan = document.getElementById("wordCount");
    const charCountSpan = document.getElementById("charCount");

    // Result Card Elements
    const resultCard = document.getElementById("resultCard");
    const resultHeading = document.getElementById("resultHeading");
    const resultIcon = document.getElementById("resultIcon");
    const confidenceValue = document.getElementById("confidenceValue");
    const realProbText = document.getElementById("realProbText");
    const fakeProbText = document.getElementById("fakeProbText");
    const realProgressBar = document.getElementById("realProgressBar");
    const fakeProgressBar = document.getElementById("fakeProgressBar");
    const explanationText = document.getElementById("explanationText");
    const disclaimerText = document.getElementById("disclaimerText");

    // Alert Banner
    const clientAlert = document.getElementById("clientAlert");
    const clientAlertMessage = document.getElementById("clientAlertMessage");

    // Sample buttons
    const sampleRealBtn = document.getElementById("sampleRealBtn");
    const sampleFakeBtn = document.getElementById("sampleFakeBtn");

    // Sample data for quick testing
    const SAMPLES = {
        real: "NASA announced on Wednesday that the James Webb Space Telescope observed atmospheric water vapor signatures on a rocky exoplanet orbiting within the habitable zone of a red dwarf star 120 light years away. Peer-reviewed spectroscopic measurements verified the presence of chemical absorption lines corresponding to H2O molecules.",
        fake: "Secret leaked government files prove that world leaders have met inside a hollow moon base with extraterrestrial reptilian overlords. An anonymous whistleblower claims that 5G frequencies are secretly spraying synthetic pathogens into city drinking supplies to force everyone into mind-control compliance."
    };

    /**
     * Updates word and character counters.
     */
    function updateCounters() {
        const text = newsInput.value.trim();
        const chars = newsInput.value.length;
        const words = text ? text.split(/\s+/).length : 0;
        wordCountSpan.textContent = words.toLocaleString();
        charCountSpan.textContent = chars.toLocaleString();
    }

    newsInput.addEventListener("input", () => {
        updateCounters();
        hideAlert();
    });

    // Initial counter sync
    updateCounters();

    /**
     * Shows a warning or error message alert.
     */
    function showAlert(msg) {
        clientAlertMessage.textContent = msg;
        clientAlert.classList.remove("d-none");
        clientAlert.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    /**
     * Hides the error alert.
     */
    function hideAlert() {
        clientAlert.classList.add("d-none");
        clientAlertMessage.textContent = "";
    }

    /**
     * Toggles the button loading state.
     */
    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnSpinner.classList.remove("d-none");
            btnIcon.classList.add("d-none");
            btnText.textContent = "Analyzing Article...";
        } else {
            analyzeBtn.disabled = false;
            btnSpinner.classList.add("d-none");
            btnIcon.classList.remove("d-none");
            btnText.textContent = "Analyze News";
        }
    }

    /**
     * Handles the prediction request.
     */
    async function analyzeNews() {
        const text = newsInput.value.trim();

        hideAlert();

        if (!text) {
            showAlert("Please enter or paste news article text before analyzing.");
            newsInput.focus();
            return;
        }

        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok || data.status === "error") {
                showAlert(data.message || "Failed to analyze the article. Please verify that the model is trained.");
                resultCard.classList.add("d-none");
                return;
            }

            renderResult(data);

        } catch (error) {
            console.error("Prediction fetch error:", error);
            showAlert("A network or server error occurred. Ensure your Flask server is running and try again.");
            resultCard.classList.add("d-none");
        } finally {
            setLoading(false);
        }
    }

    /**
     * Renders prediction verdict, colors, meters, and explanation in the Result Card.
     */
    function renderResult(data) {
        const isReal = data.is_real;

        // Reset classes
        resultCard.classList.remove("verdict-real", "verdict-fake", "d-none");

        if (isReal) {
            resultCard.classList.add("verdict-real");
            resultHeading.textContent = "REAL NEWS";
            resultIcon.innerHTML = `<i class="bi bi-shield-check"></i>`;
        } else {
            resultCard.classList.add("verdict-fake");
            resultHeading.textContent = "FAKE NEWS";
            resultIcon.innerHTML = `<i class="bi bi-shield-x"></i>`;
        }

        // Metrics
        confidenceValue.textContent = `${data.confidence}%`;
        realProbText.textContent = `${data.real_probability}%`;
        fakeProbText.textContent = `${data.fake_probability}%`;

        // Progress bar animation
        realProgressBar.style.width = `${data.real_probability}%`;
        realProgressBar.setAttribute("aria-valuenow", data.real_probability);

        fakeProgressBar.style.width = `${data.fake_probability}%`;
        fakeProgressBar.setAttribute("aria-valuenow", data.fake_probability);

        // Explanation & Disclaimer
        explanationText.textContent = data.explanation;
        if (data.disclaimer) {
            disclaimerText.textContent = data.disclaimer;
        }

        // Trigger animation
        resultCard.classList.remove("transition-fade");
        void resultCard.offsetWidth; // Force DOM reflow
        resultCard.classList.add("transition-fade");

        // Smooth scroll to results
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    /**
     * Clears input and resets results.
     */
    function clearForm() {
        newsInput.value = "";
        updateCounters();
        hideAlert();
        resultCard.classList.add("d-none");
        newsInput.focus();
    }

    // Event Listeners
    analyzeBtn.addEventListener("click", analyzeNews);
    clearBtn.addEventListener("click", clearForm);

    // Enter key shortcut (Ctrl + Enter or Cmd + Enter)
    newsInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            analyzeNews();
        }
    });

    // Sample loader events
    if (sampleRealBtn) {
        sampleRealBtn.addEventListener("click", () => {
            newsInput.value = SAMPLES.real;
            updateCounters();
            hideAlert();
            analyzeNews();
        });
    }

    if (sampleFakeBtn) {
        sampleFakeBtn.addEventListener("click", () => {
            newsInput.value = SAMPLES.fake;
            updateCounters();
            hideAlert();
            analyzeNews();
        });
    }
});
