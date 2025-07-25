document.getElementById('inputForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Get input values
    var age = parseFloat(document.getElementById('age').value);
    var calories = parseFloat(document.getElementById('calories').value);
    var sleep = parseFloat(document.getElementById('sleep').value);
    var income = parseFloat(document.getElementById('income').value);
    var workingHours = parseFloat(document.getElementById('working_hours').value);

    // Prepare data to send to the server
    var data = {
        age: age,
        calories: calories,
        sleep: sleep,
        income: income,
        working_hours: workingHours
    };

    // Send POST request to the server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById('result').innerHTML = 'Predicted Healthy Life Percent: ' + data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
