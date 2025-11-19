from flask import Flask, jsonify, request
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "running", 
        "service": "Музейная система",
        "neural_network": "disabled",
        "message": "✅ Сервер работает! Нейросеть отключена для теста"
    })

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "version": "1.0"})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Заглушка для анализа - всегда возвращает тестовый результат"""
    try:
        # Просто проверяем что изображение валидное
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image'}), 400
            
        image_data = data['image']
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # ВОЗВРАЩАЕМ ТЕСТОВЫЙ РЕЗУЛЬТАТ
        return jsonify({
            "status": "success",
            "results": [
                {
                    "similarity": 0.95,
                    "class": "test_category",
                    "description": "Тестовый объект - нейросеть отключена",
                    "image_name": "test.jpg"
                }
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug')
def debug():
    return jsonify({
        "status": "server_only",
        "message": "Нейросеть будет добавлена позже",
        "next_step": "pythonanywhere"
    })

if __name__ == '__main__':
    print("🚀 Запускаю УПРОЩЕННЫЙ сервер...")
    app.run(host='0.0.0.0', port=5000, debug=False)
