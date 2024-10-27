document.getElementById('start-button').addEventListener('click', function() {
    fetch('/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.status));
});

document.getElementById('stop-button').addEventListener('click', function() {
    fetch('/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.status));
});
