"""
מסווג אודיו מבוסס wav2vec2.

הארכיטקטורה: encoder מאומן מראש (קפוא) + attention pooling + ראש סיווג.
זוהי הגישה הסטנדרטית לכיוון עדין (fine-tuning) על מאגר נתונים קטן —
האנקודר כבר יודע להוציא מאפיינים אקוסטיים מועילים, אנחנו רק לומדים
איך לסווג אותם לארבע הקטגוריות שלנו.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class AttentionPooling(nn.Module):
    """
    Attention pooling — במקום ממוצע פשוט, המודל לומד אילו פריימים חשובים יותר.

    לדוגמה: כשנציג אנושי עונה, יש בדרך כלל כמה פריימים של דיבור ברור
    בתוך רעש רקע — attention pooling נותן להם משקל גבוה יותר.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H) — פלט האנקודר.
            mask: (B, T) — מסכה: 1 = פריים אמיתי, 0 = padding.

        Returns:
            (B, H) — ווקטור מאוגד עם משקלות attention.
        """
        scores = self.attention(hidden_states).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)    # (B, T, 1)
        return (hidden_states * weights).sum(dim=1)          # (B, H)


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

        self.pooling = AttentionPooling(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values: (batch, samples) — אודיו ב-16 קילוהרץ.
            attention_mask: (batch, samples) — מסכה: 1 = אודיו אמיתי, 0 = padding.

        Returns:
            logits: (batch, num_classes) — ציוני סיווג לפני softmax.
        """
        out = self.encoder(
            input_values, attention_mask=attention_mask,
        )
        hidden_states = out.last_hidden_state  # (B, T, H)

        # בניית מסכה ברמת הפריימים (ה-encoder מקצר את הרצף)
        frame_mask = None
        if attention_mask is not None:
            # wav2vec2 מבצע דגימת-חסר (downsampling) — נחשב את האורך בפלט
            input_lengths = attention_mask.sum(dim=1).long()
            output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
            T = hidden_states.shape[1]
            frame_mask = torch.zeros(hidden_states.shape[0], T, device=hidden_states.device)
            for i, length in enumerate(output_lengths):
                frame_mask[i, :length] = 1

        pooled = self.pooling(hidden_states, frame_mask)  # (B, H)
        return self.classifier(pooled)
