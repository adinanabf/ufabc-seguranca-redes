import os
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model-v3.pt")


def load_deepfake_model():
    """Load EfficientNet-based deepfake model and preprocessing transform.

    Returns (model, device, transform).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_features, 2),
    )

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return model, device, transform


def predict_frames(frames, model, device, transform):
    """Predict deepfake label/probabilities for a list of PIL Images.

    Returns (label, probs) where label is 0=REAL,1=FAKE and
    probs is a numpy array of shape (2,) with [p_real, p_fake].
    """
    if not frames:
        raise ValueError("No frames provided for prediction")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            inp = transform(frame).unsqueeze(0).to(device)
            out = model(inp)
            prob = torch.softmax(out, dim=1)
            all_probs.append(prob.cpu())

    avg_prob = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    predicted = int(torch.argmax(avg_prob).item())
    return predicted, avg_prob.numpy()
