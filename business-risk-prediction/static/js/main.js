// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // Initialize AOS Animation
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800,
            once: true,
            offset: 100
        });
    }

    // Load Charts on Index Page
    const chartsContainer = document.getElementById('chartsContainer');
    if (chartsContainer) {
        fetchMetricsAndDrawCharts();
    }

    // Prediction Form Submission
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(predictForm);
            const data = Object.fromEntries(formData.entries());
            
            const loading = document.getElementById('loadingOverlay');
            const resultCard = document.getElementById('resultCard');
            const formInputCard = document.getElementById('formInputCard');
            
            loading.style.display = 'flex';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                loading.style.display = 'none';
                
                if (response.ok) {
                    showResult(result);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                loading.style.display = 'none';
                alert('Request failed. Please check your connection.');
                console.error(error);
            }
        });
    }
});

function showResult(result) {
    const resultCard = document.getElementById('resultCard');
    const scoreVal = document.getElementById('scoreValue');
    const categoryVal = document.getElementById('riskCategory');
    
    scoreVal.textContent = result.score;
    categoryVal.textContent = result.category;
    
    // reset classes
    scoreVal.className = 'risk-score-display';
    if (result.category === 'Low Risk') scoreVal.classList.add('risk-low');
    else if (result.category === 'Medium Risk') scoreVal.classList.add('risk-medium');
    else scoreVal.classList.add('risk-high');
    
    resultCard.style.display = 'block';
    
    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetForm() {
    document.getElementById('predictForm').reset();
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('formInputCard').scrollIntoView({ behavior: 'smooth' });
}

async function fetchMetricsAndDrawCharts() {
    try {
        const response = await fetch('/metrics');
        if (!response.ok) {
            console.log("Metrics not available yet.");
            return;
        }
        const data = await response.json();
        
        const models = Object.keys(data);
        const accuracy = models.map(m => data[m].Accuracy || (data[m].R2 * 100).toFixed(2));
        const r2 = models.map(m => data[m].R2);
        const mae = models.map(m => data[m].MAE);
        const rmse = models.map(m => data[m].RMSE);
        
        drawChart('accuracyChart', 'Accuracy (%) (Higher is Better)', models, accuracy, 'rgba(164, 255, 164, 0.8)');
        drawChart('r2Chart', 'R2 Score (Higher is Better)', models, r2, 'rgba(255, 164, 164, 0.8)');
        drawChart('maeChart', 'Mean Absolute Error (Lower is Better)', models, mae, 'rgba(186, 223, 219, 0.8)');
        drawChart('rmseChart', 'Root Mean Squared Error (Lower is Better)', models, rmse, 'rgba(255, 189, 189, 0.8)');
        
    } catch (error) {
        console.error("Failed to load metrics:", error);
    }
}

function drawChart(canvasId, label, labels, data, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Set text color for light mode
    Chart.defaults.color = '#718096';
    Chart.defaults.borderColor = 'rgba(0, 0, 0, 0.05)';

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
