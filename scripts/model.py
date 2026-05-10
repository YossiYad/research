"""
מסווג אודיו מבוסס wav2vec2.

הארכיטקטורה: encoder מאומן מראש (קפוא) + שכבת סיווג חדשה.
זוהי הגישה הסטנדרטית לכיוון עדין (fine-tuning) על מאגר נתונים קטן —
האנקודר כבר יודע להוציא מאפיינים אקוסטיים מועילים, אנחנו רק לומדים
איך לסווג אותם לארבע הקטגוריות שלנו.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioClassifier(nn.Module):
    """
    מסווג אודיו ל-N קטגוריות.

    Args:
        base_model: שם המודל המאומן מראש מ-HuggingFace.
            ברירת מחדל: 'facebook/wav2vec2-base' (95M פרמטרים, מהיר).
            חלופות מומלצות:
              - 'facebook/wav2vec2-xls-r-300m' — תמיכה רב-לשונית כולל עברית, איטי יותר.
              - 'imvladikon/wav2vec2-xls-r-300m-hebrew' — מותאם לעברית.
        num_classes: מספר הקטגוריות (4 = human/ivr/music/recording).
        freeze_encoder: האם להקפיא את האנקודר. True = רק שכבת הסיווג מתאמנת
            (מהיר, פחות נתונים נדרשים). False = מתאמן הכל (איטי, נדרשים יותר נתונים).
        dropout: שיעור ה-dropout בשכבת הסיווג, למניעת overfitting.
    """

    def __init__(
        self,
        base_model: str = "facebook/wav2vec2-base",
        num_classes: int = 4,
        freeze_encoder: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(base_model)
        hidden_size: int = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: (batch, samples) — אודיו ב-16 קילוהרץ.

        Returns:
            logits: (batch, num_classes) — ציוני סיווג לפני softmax.
        """
        encoder_out = self.encoder(input_values).last_hidden_state  # (B, T, H)
        pooled = encoder_out.mean(dim=1)                            # (B, H)
        return self.classifier(pooled)
