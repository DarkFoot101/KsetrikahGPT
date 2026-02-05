
// --- TAB SWITCHING ---
function switchTab(tab) {
    const marketSection = document.getElementById('market-section');
    const agroSection = document.getElementById('agro-section');
    const schemesSection = document.getElementById('schemes-section');
    const updatesSection = document.getElementById('updates-section');
    const storiesSection = document.getElementById('stories-section');

    const navMarket = document.getElementById('nav-market');
    const navAgro = document.getElementById('nav-agro');
    const navSchemes = document.getElementById('nav-schemes');
    const navUpdates = document.getElementById('nav-updates');
    const navStories = document.getElementById('nav-stories');

    // Hide all
    marketSection.classList.add('hidden');
    agroSection.classList.add('hidden');
    schemesSection.classList.add('hidden');
    updatesSection.classList.add('hidden');
    if (storiesSection) storiesSection.classList.add('hidden');

    navMarket.classList.remove('active');
    navAgro.classList.remove('active');
    navSchemes.classList.remove('active');
    navUpdates.classList.remove('active');
    if (navStories) navStories.classList.remove('active');

    // Show selected
    if (tab === 'market') {
        marketSection.classList.remove('hidden');
        navMarket.classList.add('active');
    } else if (tab === 'agro') {
        agroSection.classList.remove('hidden');
        navAgro.classList.add('active');
    } else if (tab === 'schemes') {
        schemesSection.classList.remove('hidden');
        navSchemes.classList.add('active');
        if (schemesData.length === 0) { // First load
            loadMockSchemes();
            renderSchemes();
        }
    } else if (tab === 'updates') {
        updatesSection.classList.remove('hidden');
        navUpdates.classList.add('active');
        if (updatesData.length === 0) { // First load
            loadMockUpdates();
            renderUpdates();
        }
    } else if (tab === 'stories') {
        storiesSection.classList.remove('hidden');
        navStories.classList.add('active');
        if (storiesData.length === 0) {
            loadMockStories();
            renderStories();
        }
    }
}

// --- MARKET PREDICTION ---
async function predict() {
    const errorBox = document.getElementById('marketError');
    const resultBox = document.getElementById('resultBox');

    errorBox.style.display = 'none';
    resultBox.classList.add('hidden');

    // Get values
    const payload = {
        Commodity_Group: document.getElementById('grp').value,
        Commodity: document.getElementById('comm').value,
        MSP: +document.getElementById('msp').value,
        Price_1DayAgo: +document.getElementById('p1').value,
        Price_2DaysAgo: +document.getElementById('p2').value,
        Arrival_Today: +document.getElementById('a0').value,
        Arrival_1DayAgo: +document.getElementById('a1').value,
        Arrival_2DaysAgo: +document.getElementById('a2').value
    };

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Show Result
        resultBox.classList.remove('hidden');
        const priceEl = document.getElementById('price');
        const trendEl = document.getElementById('trend');

        // Count up animation for price
        let startPrice = 0;
        const endPrice = data.predicted_price_tomorrow;
        const duration = 1000;
        const startTime = performance.now();

        function animatePrice(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out quart
            const ease = 1 - Math.pow(1 - progress, 4);

            const currentVal = startPrice + (endPrice - startPrice) * ease;
            priceEl.innerText = `₹ ${currentVal.toFixed(2)}`;

            if (progress < 1) {
                requestAnimationFrame(animatePrice);
            }
        }
        requestAnimationFrame(animatePrice);

        // Update Trend
        trendEl.className = 'trend-badge ' + (data.trend === 'UP' ? 'trend-up' : 'trend-down');
        trendEl.innerHTML = data.trend === 'UP'
            ? '<iconify-icon icon="lucide:trending-up"></iconify-icon> Bullish Trend'
            : '<iconify-icon icon="lucide:trending-down"></iconify-icon> Bearish Trend';

    } catch (e) {
        errorBox.innerText = "Error: " + e.message;
        errorBox.style.display = 'block';
    }
}

// --- AI AGRONOMIST ---
let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;
let isRecording = false;

async function toggleRecording() {
    const btn = document.getElementById('recordBtn');
    const status = document.getElementById('recStatus');

    if (!isRecording) {
        // Start Recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

            mediaRecorder.onstop = () => {
                recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                status.innerText = "✅ Recording saved ready for analysis.";
                document.getElementById('voiceNote').value = ""; // Clear file input
            };

            mediaRecorder.start();
            isRecording = true;
            btn.classList.add('recording');
            btn.innerHTML = '<iconify-icon icon="lucide:square"></iconify-icon> Stop Recording';
            status.innerText = "🔴 Recording... Speak clearly about your crop issue.";
        } catch (err) {
            alert("Microphone access denied: " + err);
        }
    } else {
        // Stop Recording
        if (mediaRecorder) {
            mediaRecorder.stop();
            isRecording = false;
            btn.classList.remove('recording');
            btn.innerHTML = '<iconify-icon icon="lucide:mic"></iconify-icon> Start Recording';
        }
    }
}

async function analyzeCrop() {
    const imgInput = document.getElementById('cropImage');
    const audInput = document.getElementById('voiceNote');
    const promptRef = document.getElementById('promptText');
    const langRef = document.getElementById('responseLang');
    const btn = document.getElementById('analyzeBtn');
    const errBox = document.getElementById('agroError');
    const resBox = document.getElementById('agroResult');
    const txtBox = document.getElementById('agroText');
    const promptBox = document.getElementById('transcribedPrompt');
    const audPlayer = document.getElementById('agroAudio');

    // Reset
    errBox.style.display = 'none';
    resBox.classList.add('hidden');
    audPlayer.style.display = 'none';
    promptBox.innerText = "";

    if (!imgInput.files[0]) {
        errBox.innerText = "Please upload an image of the crop to start analysis.";
        errBox.style.display = 'block';
        return;
    }

    btn.disabled = true;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<iconify-icon icon="lucide:loader-2" class="spin"></iconify-icon> Analyzing with AI...';

    try {
        const formData = new FormData();
        formData.append('image', imgInput.files[0]);

        if (recordedBlob) {
            formData.append('audio', recordedBlob, 'recording.webm');
        } else if (audInput.files[0]) {
            formData.append('audio', audInput.files[0]);
        }

        formData.append('prompt', promptRef.value);
        formData.append('language', langRef.value);

        // 1. Analyze
        const res = await fetch('/assistant/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Analysis failed');

        // Show Text
        resBox.classList.remove('hidden');
        txtBox.innerText = data.response;
        if (data.transcribed_prompt) {
            promptBox.innerText = "User Query: " + data.transcribed_prompt;
        }

        // 2. TTS
        try {
            const ttsRes = await fetch('/assistant/text-to-speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: data.response,
                    language: langRef.value
                })
            });

            if (ttsRes.ok) {
                const blob = await ttsRes.blob();
                const url = URL.createObjectURL(blob);
                audPlayer.src = url;
                audPlayer.style.display = 'block';
                // Auto play might be blocked by browsers, but we try
                audPlayer.play().catch(e => console.log("Auto-play prevented"));
            }
        } catch (ttsErr) {
            console.warn("TTS Error:", ttsErr);
        }

    } catch (e) {
        errBox.innerText = e.message;
        errBox.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// --- MOCK DATA FOR SCHEMES & UPDATES & STORIES ---
let schemesData = [];
let updatesData = [];
let storiesData = [];

function loadMockSchemes() {
    schemesData = [
        {
            title: "PM-KISAN Samman Nidhi",
            category: "central",
            eligibility: "Small & Marginal Farmers",
            benefit: "₹6,000 per year in 3 installments",
            link: "https://pmkisan.gov.in/",
            image: "/static/img/scheme_placeholder.png", // Keep original for variety
            video: "https://www.youtube.com/watch?v=123"
        },
        {
            title: "Pradhan Mantri Fasal Bima Yojana",
            category: "central",
            eligibility: "All Farmers with Insurable Crops",
            benefit: "Comprehensive crop insurance coverage",
            link: "https://pmfby.gov.in/",
            image: "/static/img/scheme_insurance.png", // NEW
            video: "https://www.youtube.com/watch?v=456"
        },
        {
            title: "Mahatma Jyotirao Phule Shetkari Karjmukti",
            category: "maharashtra",
            eligibility: "Farmers with loans up to ₹2 Lakh",
            benefit: "Loan waiver up to ₹2 Lakhs",
            link: "#",
            image: "/static/img/scheme_placeholder.png",
            video: null
        },
        {
            title: "Punjab State Tubewell Scheme",
            category: "punjab",
            eligibility: "Farmers in Punjab",
            benefit: "Subsidized electricity for tubewells",
            link: "#",
            image: "/static/img/scheme_irrigation.png", // NEW
            video: null
        },
        {
            title: "Kisan Credit Card (KCC)",
            category: "central",
            eligibility: "All Farmers, Tenant Farmers",
            benefit: "Credit limit based on land holding",
            link: "#",
            image: "/static/img/scheme_placeholder.png",
            video: "https://www.youtube.com/watch?v=789"
        }
    ];
}

function loadMockStories() {
    storiesData = [
        {
            name: "Ram Lal Singh",
            region: "Ferozepur, Punjab",
            quote: "Using the market price predictor I saved my wheat harvest for 2 days and got a 15% better rate at the mandi. This technology really helps!",
            avatar: "/static/img/farmer_punjab.png" // NEW
        },
        {
            name: "Sunita Deshmukh",
            region: "Nashik, Maharashtra",
            quote: "The AI Agronomist helped me identify a fungal infection in my onion crop just by uploading a photo. It saved my entire season's yield.",
            avatar: "/static/img/farmer_maharashtra.png" // NEW
        },
        {
            name: "Gurpreet Singh",
            region: "Bhatinda, Punjab",
            quote: "Information about government schemes used to be hard to find. Now I get updates right here on my phone. Very useful app.",
            avatar: "/static/img/farmer_avatar.png" // Keep original generic one
        }
    ];
}

function loadMockUpdates() {
    updatesData = [
        {
            date: "Oct 24, 2025",
            title: "PM-KISAN 16th Installment Released",
            description: "The 16th installment of PM-KISAN has been credited to beneficiary accounts. Check status on the portal.",
            urgent: false
        },
        {
            date: "Oct 22, 2025",
            title: "Heavy Rainfall Alert: Punjab & Haryana",
            description: "IMD predicts heavy showers in northern districts. Farmers are advised to delay harvesting by 2 days.",
            urgent: true
        },
        {
            date: "Oct 18, 2025",
            title: "New MSP Rates Announced for Rabbi Crops",
            description: "Cabinet approves increase in MSP for Wheat and Mustard for the 2026-27 marketing season.",
            urgent: false
        },
        {
            date: "Oct 10, 2025",
            title: "Subsidy for Solar Pumps Increased",
            description: "Government increases subsidy for PM-KUSUM solar pump installation to 60%.",
            urgent: false
        }
    ];
}

function renderSchemes(filter = 'all') {
    const grid = document.getElementById('schemes-grid');
    grid.innerHTML = "";

    let delay = 0;

    schemesData.forEach(scheme => {
        if (filter !== 'all' && scheme.category !== filter) return;

        const badgeClass = scheme.category === 'central' ? 'badge-central' : 'badge-state';
        const badgeText = scheme.category === 'central' ? 'Central Govt' : scheme.category.toUpperCase();

        const videoBtn = scheme.video
            ? `<a href="${scheme.video}" target="_blank" class="video-btn"><iconify-icon icon="lucide:play-circle"></iconify-icon> Watch Video</a>`
            : '';

        const card = document.createElement('div');
        card.className = 'scheme-card';
        card.style.animation = `fadeInUp 0.5s ease forwards ${delay}s`;
        card.style.opacity = '0';

        card.innerHTML = `
            <div class="card-media">
                <img src="${scheme.image}" alt="${scheme.title}">
                ${videoBtn}
            </div>
            <div class="scheme-header">
                <div class="scheme-title">${scheme.title}</div>
            </div>
            <div style="margin-bottom: 12px;">
                 <span class="badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="scheme-body">
                <div class="scheme-info">
                    <div class="info-label">Eligibility</div>
                    <div class="info-text">${scheme.eligibility}</div>
                </div>
                <div class="scheme-info">
                    <div class="info-label">Benefits</div>
                    <div class="info-text" style="color: var(--primary); font-weight: 500;">${scheme.benefit}</div>
                </div>
            </div>
            <a href="${scheme.link}" target="_blank" class="btn-sm">
                View Details <iconify-icon icon="lucide:arrow-right" style="vertical-align: middle;"></iconify-icon>
            </a>
        `;

        grid.appendChild(card);
        delay += 0.1;
    });
}

function renderStories() {
    const grid = document.getElementById('stories-grid');
    grid.innerHTML = "";

    let delay = 0;

    storiesData.forEach(story => {
        const card = document.createElement('div');
        card.className = 'story-card';
        card.style.animation = `fadeInUp 0.5s ease forwards ${delay}s`;
        card.style.opacity = '0';

        card.innerHTML = `
            <iconify-icon icon="lucide:quote" class="quote-icon"></iconify-icon>
            <div class="story-header">
                <img src="${story.avatar}" class="farmer-avatar" alt="${story.name}">
                <div class="farmer-info">
                    <h3>${story.name}</h3>
                    <div class="farmer-region">${story.region}</div>
                </div>
            </div>
            <div class="story-quote">"${story.quote}"</div>
        `;
        grid.appendChild(card);
        delay += 0.1;
    });
}

function filterSchemes() {
    const filterVal = document.getElementById('schemeFilter').value;
    renderSchemes(filterVal);
}

function renderUpdates() {
    const container = document.getElementById('updates-feed');
    container.innerHTML = "";

    let delay = 0;

    updatesData.forEach(update => {
        const item = document.createElement('div');
        item.className = 'update-item';
        item.style.animation = `slideIn 0.5s ease forwards ${delay}s`;
        item.style.opacity = '0';

        const urgentClass = update.urgent ? 'urgent' : '';

        item.innerHTML = `
            <div class="update-marker ${urgentClass}"></div>
            <div class="update-date">${update.date}</div>
            <div class="update-title">${update.title}</div>
            <div class="update-desc">${update.description}</div>
        `;

        container.appendChild(item);
        delay += 0.15;
    });
}
