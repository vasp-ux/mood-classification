function predictEmotion() {
    const text = document.getElementById("textInput").value;

    if (text.length < 3) {
        alert("Please enter meaningful text");
        return;
    }

    fetch("http://127.0.0.1:5000/predict-text", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            "🧠 Detected Emotion: " + data.emotion;
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Backend not reachable");
    });
}
