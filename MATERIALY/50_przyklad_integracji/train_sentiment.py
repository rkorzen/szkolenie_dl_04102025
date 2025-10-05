# train_sentiment.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# --- dane ---
data = [
    ("I love this product", 1),
    ("It is very good", 1),
    ("This is amazing", 1),
    ("Great experience", 1),
    ("Fantastic work", 1),
    ("I really enjoy this", 1),
    ("Best thing ever", 1),
    ("Good work", 1),
    ("Absolutely wonderful", 1),
    ("Superb quality", 1),
    ("Very nice and clean", 1),
    ("This made me happy", 1),
    ("Excellent choice", 1),
    ("Highly recommended", 1),
    ("I am very satisfied", 1),
    ("So impressive", 1),
    ("Positive experience overall", 1),
    ("I would buy again", 1),
    ("Really good value", 1),
    ("Amazing service", 1),
    ("Very good", 1),
    ("Very interesting", 1),

    ("I hate it", 0),
    ("This is terrible", 0),
    ("Worst ever", 0),
    ("I dislike this", 0),
    ("Awful experience", 0),
    ("Not good at all", 0),
    ("Very bad", 0),
    ("Not good work", 0),
    ("Horrible service", 0),
    ("Extremely disappointing", 0),
    ("Really poor quality", 0),
    ("A complete waste of money", 0),
    ("Not worth it", 0),
    ("This made me angry", 0),
    ("Totally unacceptable", 0),
    ("Would not recommend", 0),
    ("Very frustrating", 0),
    ("Negative experience overall", 0),
    ("I regret buying this", 0),
    ("Awful customer support", 0),
    ("Dissapointed", 0),
    ("Don't like it", 0),
    ("Very boring", 0),
    ("this is so bad", 0),
]

texts, labels = zip(*data)
labels = torch.tensor(labels, dtype=torch.long)

# --- tokenizacja ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
X = torch.tensor(X, dtype=torch.float32)

# --- split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.1, stratify=labels, random_state=42
)

# --- model: prosta regresja logistyczna ---
class SentimentNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

model = SentimentNet(input_dim=X.shape[1], num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- trening ---
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {acc:.2f}")

# --- ewaluacja końcowa ---
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
print("Final Accuracy:", acc)

# --- zapis ---
torch.save(model.state_dict(), "sentiment_model.pth")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model i vectorizer zapisane")
