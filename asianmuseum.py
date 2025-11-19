import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Для облака используем относительные пути
script_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к папкам с данными (относительные)
dataset_path = os.path.join(script_dir, 'MyDataset', 'path_to_images')
descriptions_path = os.path.join(script_dir, 'MyDataset', 'path_to_descriptions')

# Создаем папки если их нет
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(descriptions_path, exist_ok=True)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class SimpleResNet18(nn.Module):
    def __init__(self):
        super(SimpleResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        return self.model(x)

class MuseumSystem:
    def __init__(self):
        self.feature_extractor = SimpleResNet18().to(device)
        self.feature_extractor.eval()
        self.database = []
        self.features = None
        
    def build_database(self):
        print("Загружаю базу музея...")
        
        # Проверяем существование папок
        if not os.path.exists(dataset_path):
            print(f"ОШИБКА: Папка с изображениями не найдена: {dataset_path}")
            return False
        if not os.path.exists(descriptions_path):
            print(f"ОШИБКА: Папка с описаниями не найдена: {descriptions_path}")
            return False
        
        for class_name in sorted(os.listdir(dataset_path)):
            class_image_dir = os.path.join(dataset_path, class_name)
            class_desc_dir = os.path.join(descriptions_path, class_name)
            
            if not os.path.isdir(class_image_dir):
                continue
                
            for image_file in os.listdir(class_image_dir):
                if image_file.lower().endswith('.jpg'):
                    image_path = os.path.join(class_image_dir, image_file)
                    
                    txt_file = os.path.splitext(image_file)[0] + '.txt'
                    txt_path = os.path.join(class_desc_dir, txt_file)
                    
                    description = "Нет описания"
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                description = f.read().strip()
                        except:
                            description = f"Объект {class_name}"
                    else:
                        description = f"Объект {class_name}"
                    
                    self.database.append({
                        'image_path': image_path,
                        'description': description,
                        'class': class_name
                    })
        
        print(f"База построена: {len(self.database)} экспонатов")
        
        self.features = []
        for i, item in enumerate(self.database):
            try:
                image = Image.open(item['image_path']).convert('RGB')
                tensor = data_transforms(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = self.feature_extractor(tensor)
                    features = features.cpu().numpy().flatten()
                
                self.features.append(features)
                print(f"Обработано {i+1}/{len(self.database)} изображений")
            except Exception as e:
                print(f"Ошибка обработки {item['image_path']}: {e}")
                self.features.append(np.zeros(512))
        
        self.features = np.array(self.features)
        print("Признаки извлечены")
        return True
    
    def search_image(self, image):
        """Ищет похожие изображения в базе"""
        if not self.database or self.features is None:
            return {"status": "error", "message": "База данных не загружена"}
        
        try:
            # Преобразуем изображение для нейросети
            image_rgb = image.convert('RGB')
            tensor = data_transforms(image_rgb).unsqueeze(0).to(device)
            
            # Извлекаем признаки
            with torch.no_grad():
                user_features = self.feature_extractor(tensor)
                user_features = user_features.cpu().numpy().flatten()
            
            # Сравниваем с базой
            similarities = cosine_similarity([user_features], self.features)[0]
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_item = self.database[best_idx]
            
            # Возвращаем результат в формате JSON
            return {
                "status": "success",
                "similarity": float(best_similarity),
                "class": best_item['class'],
                "description": best_item['description'],
                "image_path": best_item['image_path']
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}