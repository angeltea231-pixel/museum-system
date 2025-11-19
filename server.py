from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def home():
    return "Сервер работает! Нейросеть отключена"

if __name__ == '__main__':
    app.run(debug=False)
