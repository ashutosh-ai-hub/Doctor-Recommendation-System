let currentBooking = {}; 

document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const symptomInput = document.getElementById('symptom-input');
    const locationFilter = document.getElementById('location-filter');
    
    // Fetch and populate locations
    fetchLocations();

    // Add Enter key support for main search
    symptomInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeBtn.click();
    });

    analyzeBtn.addEventListener('click', async () => {
        const symptoms = symptomInput.value.trim();
        const selectedLocation = locationFilter.value;

        if (symptoms.length < 3) { 
            alert("Please describe your symptoms in more detail."); 
            return; 
        }

        analyzeBtn.innerText = "Analyzing...";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    symptoms: symptoms,
                    location: selectedLocation
                })
            });
            const data = await response.json();
            
            if (response.status === 200) { 
                displayResults(data); 
            } else {
                alert(data.error || "An error occurred while analyzing symptoms.");
            }
        } catch (error) { 
            alert("Server Error. Please make sure the backend is running."); 
        } finally { 
            analyzeBtn.innerText = "Analyze Symptoms"; 
            analyzeBtn.disabled = false; 
        }
    });
});

async function fetchLocations() {
    try {
        const response = await fetch('/api/locations');
        const data = await response.json();
        const locationFilter = document.getElementById('location-filter');
        
        if (data.locations) {
            data.locations.forEach(loc => {
                const option = document.createElement('option');
                option.value = loc;
                option.textContent = loc;
                locationFilter.appendChild(option);
            });
        }
    } catch (error) {
        console.error("Error fetching locations:", error);
    }
}

window.displayResults = function(data) {
    const resultSection = document.getElementById('result-section');
    const doctorGrid = document.getElementById('doctor-grid');
    const specialistSpan = document.getElementById('predicted-specialist');
    
    // Show predicted disease if available
    let specialistText = data.specialization;
    if (data.disease) {
        specialistText += ` (Likely: ${data.disease})`;
    }
    specialistSpan.innerText = specialistText;
    doctorGrid.innerHTML = '';
    resultSection.classList.remove('hidden');
    
    const hasNearby = data.doctors && data.doctors.length > 0;
    const hasOthers = data.other_suggestions && data.other_suggestions.length > 0;

    if (!hasNearby && !hasOthers) {
        doctorGrid.innerHTML = '<p style="grid-column: 1/-1; color: var(--text-muted); text-align: center;">No doctors found for this specialization.</p>';
        return;
    }

    // Show Nearby Doctors
    if (hasNearby) {
        if (data.searched_location !== 'All') {
            const header = document.createElement('h3');
            header.className = 'grid-header';
            header.innerHTML = `📍 Doctors in ${data.searched_location}`;
            header.style.gridColumn = "1/-1";
            doctorGrid.appendChild(header);
        }

        data.doctors.forEach((doc, index) => {
            renderDoctorCard(doc, index, doctorGrid);
        });
    }

    // Show Suggestions (if location filter was used)
    if (hasOthers) {
        const header = document.createElement('h3');
        header.className = 'grid-header';
        header.innerHTML = hasNearby ? `💡 Suggested Doctors in Other Locations` : `🔍 No doctors found in ${data.searched_location}. Try these specialists:`;
        header.style.gridColumn = "1/-1";
        header.style.marginTop = "2rem";
        doctorGrid.appendChild(header);

        data.other_suggestions.forEach((doc, index) => {
            renderDoctorCard(doc, index + (hasNearby ? data.doctors.length : 0), doctorGrid);
        });
    }
    
    // Scroll smoothly to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderDoctorCard(doc, index, container) {
    const card = document.createElement('div');
    card.className = 'doctor-card';
    card.style.animation = `fadeIn 0.5s ease-out ${index * 0.1}s backwards`;
    card.innerHTML = `
        <span class="badge">${doc.specialization}</span>
        <h3>${doc.name}</h3>
        <div class="stats">⭐ ${doc.rating} • 💼 ${doc.experience}y Exp.</div>
        <p class="location">📍 ${doc.location}</p>
        <button onclick="confirmBooking('${doc.name}', '${doc.specialization}')" class="btn-booking">Book Appointment</button>
    `;
    container.appendChild(card);
}

// Modal & Booking Logic
window.confirmBooking = function(docName, spec) {
    const id = "APT-" + Math.floor(10000 + Math.random() * 90000);
    currentBooking = { docName, id, spec, date: new Date().toLocaleDateString() };
    document.getElementById('modal-msg').innerText = `Appointment request successfully sent to ${docName}.`;
    document.getElementById('apt-id').innerText = id;
    document.getElementById('success-modal').classList.remove('hidden');
}

window.closeModal = function() { 
    document.getElementById('success-modal').classList.add('hidden'); 
}

// PDF Download Logic
document.getElementById('download-btn').addEventListener('click', () => {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    doc.setFillColor(2, 132, 199); // Primary color
    doc.rect(0, 0, 210, 40, 'F');
    
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("SMART-HEALTH", 20, 25);
    
    doc.setTextColor(30, 41, 59);
    doc.setFontSize(16);
    doc.text("Appointment Receipt", 20, 60);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    doc.text(`Reference ID: ${currentBooking.id}`, 20, 80);
    doc.text(`Doctor Name: ${currentBooking.docName}`, 20, 95);
    doc.text(`Specialization: ${currentBooking.spec}`, 20, 110);
    doc.text(`Date Booked: ${currentBooking.date}`, 20, 125);
    
    doc.setDrawColor(226, 232, 240);
    doc.line(20, 140, 190, 140);
    
    doc.setFontSize(10);
    doc.setTextColor(100, 116, 139);
    doc.text("Please show this receipt at the reception desk upon arrival.", 20, 155);
    
    doc.save(`Receipt_${currentBooking.id}.pdf`);
});
