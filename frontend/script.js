function sendQuery() {
    const query = document.getElementById('userQuery').value;
    fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        const resDiv = document.getElementById('response');
        if (data.error) {
            resDiv.innerText = `Error: ${data.error}`;
        } else {
            resDiv.innerHTML = `<strong>Category:</strong> ${data.category}<br><strong>Recommended Phones:</strong> ${data.recommendations.join(', ')}`;
        }
    })
    .catch(err => {
        console.error(err);
    });
}
