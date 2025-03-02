import subprocess
from flask import Flask, jsonify, render_template

app = Flask(__name__)

running_process = None  # Store the running process

def run_script():
    global running_process
    script_path = "recipe1.py"  # Make sure this path is correct
    running_process = subprocess.Popen(["python", script_path])  # Start the script

@app.route('/')
def home():
    return render_template("index.html")  # Serve the HTML file

@app.route('/start', methods=['POST'])
def start_script():
    global running_process
    if running_process is None:
        run_script()
        return jsonify({"message": "Recipe bot started"})
    return jsonify({"message": "Recipe bot is already running"})

@app.route('/stop', methods=['POST'])
def stop_script():
    global running_process
    if running_process:
        running_process.terminate()  # Stop the script
        running_process = None
        return jsonify({"message": "Recipe bot stopped"})
    return jsonify({"message": "No script is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

