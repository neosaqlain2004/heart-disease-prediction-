document.getElementById('predictBtn').addEventListener('click', function() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = {};
    let isValid = true;
    for (let [key, value] of formData.entries()) {
        if (value === '') {
            isValid = false;
            break;
        }
        data[key] = value;
    }

    if (!isValid) {
        document.getElementById('result').innerHTML = 'Please fill all fields.';
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const resultDiv = document.getElementById('result');

        // Display message based on prediction and include percentage
        let message = '';
        if (result.ultimate && result.ultimate.prediction === 'Yes') {
            message = `<p>Our analysis indicates a potential risk of heart disease (${(result.ultimate.probability * 100).toFixed(1)}%). All risks are possible, and we're telling you that at this age, this could happen to you. Please consult a healthcare professional for a detailed evaluation and medical advice.</p>`;
        } else if (result.ultimate && result.ultimate.prediction === 'No') {
            message = `<p>Our analysis indicates no significant risk of heart disease at this time (${(result.ultimate.probability * 100).toFixed(1)}%). However, maintaining a healthy lifestyle and regular check-ups is strongly recommended.</p>`;
        } else {
            message = '<p>Unable to determine prediction.</p>';
        }

        resultDiv.innerHTML = message;

        // Show individual model predictions in details section
        const detailsContent = document.getElementById('detailsContent');
        detailsContent.innerHTML = '';
        for (const model in result) {
            if (model === 'ultimate' || model === 'recommendations' || model === 'note') continue;
            const pred = result[model].prediction;
            const probModel = (result[model].probability * 100).toFixed(2);
            detailsContent.innerHTML += '<div><strong>' + model.toUpperCase() + ':</strong> ' + pred + ' (' + probModel + '%)</div>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'An error occurred.';
    });
});
