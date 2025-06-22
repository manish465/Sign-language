import cv2
import torch

from torchvision import transforms
from model import SignLanguageCNN

model = SignLanguageCNN()
model.load_state_dict(torch.load("models/model_epoch_10.pth"))
model.eval()

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "delete", "nothing"]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

cap = cv2.VideoCapture(0)

window_name = "Sign Language Prediction"
cv2.namedWindow(window_name)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    roi = frame[100:300, 100:300]

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    img = transform(roi_rgb)
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    img_tensor = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = classes[int(pred.item())]
    
    cv2.putText(frame, label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()