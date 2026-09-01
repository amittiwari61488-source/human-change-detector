// Initialize map centered on India
const map = L.map('map').setView([20.5937, 78.9629], 5);

// Add OpenStreetMap tiles (Satellite tiles can be added here)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

let currentMarker = null;
let selectedLat = null;
let selectedLon = null;

const scanBtn = document.getElementById('scan-btn');
const coordsText = document.getElementById('coords');
const loader = document.getElementById('loader');
const imagesContainer = document.getElementById('images-container');
const resultMask = document.getElementById('result-mask');

// Handle Map Clicks
map.on('click', function(e) {
    selectedLat = e.latlng.lat;
    selectedLon = e.latlng.lng;
    
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }
    currentMarker = L.marker([selectedLat, selectedLon]).addTo(map);
    
    coordsText.textContent = `${selectedLat.toFixed(4)}, ${selectedLon.toFixed(4)}`;
    scanBtn.disabled = false;
});

// Handle Scan Button
scanBtn.addEventListener('click', async () => {
    if (!selectedLat || !selectedLon) return;
    
    scanBtn.disabled = true;
    loader.classList.remove('hidden');
    imagesContainer.classList.add('hidden');

    try {
        const response = await fetch('/api/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat: selectedLat, lon: selectedLon })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            resultMask.src = data.mask_url;
            imagesContainer.classList.remove('hidden');
        }
    } catch (error) {
        alert("Error scanning region.");
        console.error(error);
    } finally {
        scanBtn.disabled = false;
        loader.classList.add('hidden');
    }
});