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
            resDiv.innerHTML = `<span style="color:red;">❌ Error: ${data.error}</span>`;
        } else if (data.recommendations.length === 0) {
            resDiv.innerHTML = `<span style="color:orange;">⚠️ No matching phones found.</span>`;
        } else {
            let html = `<strong>🔍 Category:</strong> ${data.category}<br><br>`;
            html += `<strong>📱 Recommended Phones:</strong><ul>`;
            data.recommendations.forEach(phone => {
                html += `
                    <li>
                        <strong>${phone.name}</strong><br>
                        📸 Camera: ${phone.camera_quality} |
                        🔋 Battery: ${phone.battery_life} |
                        ⚡ Performance: ${phone.performance} |
                        💰 Price: Rs. ${phone.price}
                    </li><hr>
                `;
            });
            html += `</ul>`;
            resDiv.innerHTML = html;
        }
    })
    .catch(err => {
        console.error(err);
        document.getElementById('response').innerText = '⚠️ Something went wrong. Check console.';
    });
}
