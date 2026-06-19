"""
MOPSO-Based CNN for Keyword Selection on Google Ads
keyword_model.py — CNN model definition

B.E. Final Year Project, AIT Chikkamagaluru (VTU), 2020-21
Award: Best Project of the Year — IISc Bangalore 44th Student Project Programme
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


class KeywordCNN:
    """
    Improved CNN for keyword classification.

    Extends a standard text CNN by incorporating both linguistic
    features from word embeddings and contextual keyword features —
    enabling accurate positive/negative keyword classification for
    Google Ads campaigns.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        max_seq_len: int = 50,
        num_classes: int = 4,
        num_filters: int = 128,
        kernel_sizes: list = [2, 3, 4],
        dropout_rate: float = 0.5,
        embedding_matrix: np.ndarray = None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        self.model = self._build()

    def _build(self) -> Model:
        """Build the improved CNN architecture."""
        inputs = Input(shape=(self.max_seq_len,), name='keyword_input')

        # Embedding layer — use pretrained GloVe if available
        if self.embedding_matrix is not None:
            x = Embedding(
                self.vocab_size,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=False,
                name='glove_embedding'
            )(inputs)
        else:
            x = Embedding(
                self.vocab_size,
                self.embedding_dim,
                name='learned_embedding'
            )(inputs)

        # Parallel convolution branches with different kernel sizes
        # Captures n-gram features at multiple scales
        conv_branches = []
        for k in self.kernel_sizes:
            conv = Conv1D(
                filters=self.num_filters,
                kernel_size=k,
                activation='relu',
                name=f'conv_{k}gram'
            )(x)
            pool = GlobalMaxPooling1D(name=f'pool_{k}gram')(conv)
            conv_branches.append(pool)

        # Merge all branches
        merged = Concatenate(name='merge_branches')(conv_branches)
        merged = BatchNormalization(name='batch_norm')(merged)

        # Classification head
        dense = Dense(256, activation='relu', name='dense_1')(merged)
        drop = Dropout(self.dropout_rate, name='dropout')(dense)
        dense2 = Dense(128, activation='relu', name='dense_2')(drop)

        outputs = Dense(
            self.num_classes,
            activation='softmax',
            name='keyword_classification'
        )(dense2)

        model = Model(inputs=inputs, outputs=outputs, name='KeywordCNN')
        return model

    def compile(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs: int = 50,
        batch_size: int = 32,
        checkpoint_path: str = 'weights.{epoch:03d}-{val_accuracy:.4f}.hdf5'
    ):
        """
        Train the CNN with model checkpointing.
        Best model is saved automatically when validation accuracy improves.
        """
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict_keyword(self, keyword_vector: np.ndarray, class_labels: list) -> dict:
        """
        Predict whether a keyword is positive or negative for an ad campaign.

        Returns dict with predicted domain, confidence, and ranked keywords.
        """
        probs = self.model.predict(keyword_vector, verbose=0)[0]
        predicted_class = np.argmax(probs)
        confidence = float(np.max(probs)) * 100

        return {
            'domain': class_labels[predicted_class],
            'confidence': f'{confidence:.2f}%',
            'class_probabilities': {
                class_labels[i]: f'{probs[i]*100:.2f}%'
                for i in range(len(class_labels))
            }
        }

    def summary(self):
        self.model.summary()


if __name__ == '__main__':
    # Quick architecture check
    model = KeywordCNN(
        vocab_size=10000,
        embedding_dim=100,
        max_seq_len=50,
        num_classes=4
    )
    model.compile()
    model.summary()
