from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Constantes
MODEL_PATH = "app/models/tut4-model_100_SS_epochs_94%.pt"  # Ruta al archivo del modelo
CLASS_NAMES = ["Ampolla", "Mancha", "Pústula", "Roncha"]

class ResNetWithFeatures(nn.Module):
    """
    Modelo ResNet personalizado que devuelve tanto predicciones como representaciones de características.
    Envuelve un modelo ResNet preentrenado y añade dropout para regularización.
    """
    def __init__(self, resnet_model, dropout_p=0.7, output_dim=4):
        super().__init__()
        self.resnet = resnet_model
        
        # Extrae las capas de extracción de características de ResNet
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        self.avgpool = self.resnet.avgpool
        
        # Capa FC personalizada con Dropout
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.resnet.fc.in_features, output_dim)
        )
        
    def forward(self, x):
        """Paso hacia adelante que devuelve tanto predicciones como características"""
        x = self.features(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        x = self.fc(h)
        return x, h

def process_image(image_bytes: bytes) -> dict:
    """
    Procesa una imagen en formato de bytes, realiza una predicción usando un modelo preentrenado
    y genera una imagen con la predicción principal y el top 3 de clases.
    
    Args:
        image_bytes (bytes): Bytes de la imagen a procesar.
        
    Returns:
        dict: Diccionario con información de la predicción y la imagen generada.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar y configurar el modelo
    base_model = models.resnet50(pretrained=False)
    num_classes = len(CLASS_NAMES)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, num_classes)
    model = ResNetWithFeatures(base_model, dropout_p=0.7, output_dim=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return {"error": f"Error al cargar el modelo: {e}"}
    
    model.eval()
    
    # Procesar la imagen
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Transformaciones
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convertir la imagen a tensor
        img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return {"error": f"Error al procesar la imagen: {e}"}
    
    # Realizar predicción
    with torch.no_grad():
        outputs, _ = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy().flatten()
    
    # Obtener el top 3
    top3_idx = probabilities.argsort()[-3:][::-1]
    top3_probs = probabilities[top3_idx]
    top3_classes = [CLASS_NAMES[i] for i in top3_idx]
    
    # Preparar datos de salida
    top3_info = [{"class": cls, "probability": f"{prob * 100:.2f}%"} for cls, prob in zip(top3_classes, top3_probs)]
    predicted_class = top3_classes[0]
    confidence = f"{top3_probs[0] * 100:.2f}%"
    
    # Crear una imagen de salida con matplotlib
    buffer = io.BytesIO()
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(predicted_class, fontsize=18, fontweight="bold")
    top3_line = " | ".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(top3_classes, top3_probs)])
    plt.figtext(0.5, 0.01, top3_line, wrap=True, horizontalalignment='center', fontsize=12, color="black")
    plt.savefig(buffer, format="PNG", bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    processed_image = buffer.getvalue()
    
    return {
        "top_3": top3_info,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "processed_image": processed_image
    }
