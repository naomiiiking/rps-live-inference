from torch import nn

class RPSClassifier(nn.Module):
    """Model class with construction and forward function. Flattens vector, creates first layer, applies activation, returns logits"""
    def __init__(self, num_classes=3):
        super().__init__()
        # Convolutional network to get larger area over time
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits
        