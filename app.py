import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from model import BreastCancerMultimodalNet

DEVICE = torch.device("cpu")

print("Loading model...")
model = BreastCancerMultimodalNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image, report_text):
    if image is None:
        return "Please upload an image.", "", "", ""
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    if not report_text or not report_text.strip():
        report_text = "Breast tissue biopsy. H&E stained histopathological slide."
    enc = tokenizer(report_text, padding="max_length", max_length=128,
                    truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits, alpha, _ = model(img_tensor, input_ids, attention_mask)
        probs     = F.softmax(logits, dim=-1)
        pred      = logits.argmax(1).item()
        conf      = probs[0][pred].item()
        alpha_val = alpha[0][0].item()
    label = "MALIGNANT" if pred == 1 else "BENIGN"
    icon  = "🔴" if pred == 1 else "🟢"
    return (
        f"{icon}  {label}  —  Confidence: {conf*100:.1f}%",
        f"Benign: {probs[0][0]*100:.1f}%\nMalignant: {probs[0][1]*100:.1f}%",
        f"Image trust: {alpha_val*100:.1f}%\nText trust: {(1-alpha_val)*100:.1f}%",
        f"Alpha = {alpha_val:.3f}"
    )

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Histology Image"),
        gr.Textbox(lines=4, label="Clinical Report (optional)"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probabilities"),
        gr.Textbox(label="Modality Trust"),
        gr.Textbox(label="Alpha Gate"),
    ],
    title="Breast Cancer Multimodal Classifier",
    description="Cross-Modal Attention + Dynamic Gated Fusion | EfficientNet-B4 + BioBERT",
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
