import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from asianmuseum import MuseumSystem
from PIL import Image

app = Flask(__name__)
CORS(app)

# Инициализация системы
museum_system = MuseumSystem()

@app.before_first_request
def initialize_system():
    """Инициализация при первом запросе"""
    print("🔄 Инициализация музейной системы...")
    if museum_system.build_database():
        print("✅ База данных загружена")
    else:
        print("❌ Ошибка загрузки базы данных")

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "service": "Музейная система атрибуции",
        "version": "1.0"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'database_size': len(museum_system.database),
        'message': 'Система готова к работе'
    })

@app.route('/api/database', methods=['GET'])
def get_database_info():
    """Информация о базе данных"""
    classes = list(set(item['class'] for item in museum_system.database))
    return jsonify({
        'total_items': len(museum_system.database),
        'classes': classes,
        'status': 'loaded' if museum_system.database else 'empty'
    })

@app.route('/api/setup', methods=['POST'])
def setup_database():
    """Принудительная перезагрузка базы данных"""
    try:
        if museum_system.build_database():
            return jsonify({
                'status': 'success',
                'message': f'База данных перезагружена: {len(museum_system.database)} экспонатов'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Не удалось загрузить базу данных'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка: {str(e)}'
        })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Нет изображения в запросе'})
        
        # Проверяем, инициализирована ли система
        if not museum_system.database:
            return jsonify({'status': 'error', 'message': 'База данных не загружена'})
        
        # Декодируем base64
        if data['image'].startswith('data:'):
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Используем поиск из asianmuseum.py
        result = museum_system.search_image(image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Ошибка обработки: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Запуск сервера на порту {port}...")
    
    # Загружаем базу данных при запуске
    if museum_system.build_database():
        print("✅ База данных загружена")
        print(f"🌐 Сервер запускается на порту {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Не удалось загрузить базу данных")