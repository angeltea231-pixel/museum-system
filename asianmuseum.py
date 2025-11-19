import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from encryption import encryptor  # Импортируем шифратор
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(script_dir, 'MyDataset', 'path_to_images')
descriptions_path = os.path.join(script_dir, 'MyDataset', 'path_to_descriptions')

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
        self.encrypted_images = {}  # Храним зашифрованные изображения
        
    def build_database(self):
        print("🔒 Загружаю и шифрую базу музея...")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Папка не найдена: {dataset_path}")
            return False
            
        for class_name in sorted(os.listdir(dataset_path)):
            class_image_dir = os.path.join(dataset_path, class_name)
            
            if not os.path.isdir(class_image_dir):
                continue
                
            for image_file in os.listdir(class_image_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_image_dir, image_file)
                    
                    # Шифруем изображение
                    encrypted_data = encryptor.encrypt_image(image_path)
                    if encrypted_data is None:
                        continue
                    
                    # Сохраняем зашифрованные данные
                    image_id = f"{class_name}_{image_file}"
                    self.encrypted_images[image_id] = encrypted_data
                    
                    # Описание
                    description = f"Объект {class_name}"
                    txt_file = os.path.splitext(image_file)[0] + '.txt'
                    txt_path = os.path.join(descriptions_path, class_name, txt_file)
                    
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                description = f.read().strip()
                        except:
                            pass
                    
                    self.database.append({
                        'image_id': image_id,
                        'description': description,
                        'class': class_name,
                        'original_name': image_file
                    })
        
        print(f"📊 База зашифрована: {len(self.database)} экспонатов")
        
        # Извлекаем признаки из зашифрованных изображений
        self.features = []
        for item in self.database:
            try:
                # Дешифруем для обработки нейросетью
                encrypted_data = self.encrypted_images[item['image_id']]
                decrypted_data = encryptor.decrypt_image(encrypted_data)
                
                # Создаем изображение из байтов
                image = Image.open(io.BytesIO(decrypted_data)).convert('RGB')
                tensor = data_transforms(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = self.feature_extractor(tensor)
                    features = features.cpu().numpy().flatten()
                
                self.features.append(features)
                
            except Exception as e:
                print(f"❌ Ошибка обработки {item['image_id']}: {e}")
                self.features.append(np.zeros(512))
        
        self.features = np.array(self.features)
        print("✅ Признаки извлечены из зашифрованных изображений")
        return True
    
    def search_image(self, image):
        if not self.database or self.features is None:
            return {"status": "error", "message": "База не загружена"}
        
        try:
            image_rgb = image.convert('RGB')
            tensor = data_transforms(image_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                user_features = self.feature_extractor(tensor)
                user_features = user_features.cpu().numpy().flatten()
            
            similarities = cosine_similarity([user_features], self.features)[0]
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append({
                        "similarity": float(similarities[idx]),
                        "class": self.database[idx]['class'],
                        "description": self.database[idx]['description'],
                        "image_name": self.database[idx]['original_name']
                    })
            
            if results:
                return {"status": "success", "results": results}
            else:
                return {"status": "error", "message": "Не найдено похожих"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
