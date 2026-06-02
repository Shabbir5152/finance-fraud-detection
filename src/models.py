from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Supervised Approach: XGBoost
def train_xgboost(X_train, y_train):
    # n_jobs=-1 uses all cores of your Ultra 7 for faster training
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Unsupervised Approach: Autoencoder
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(14, activation="tanh")(input_layer)
    encoder = Dense(7, activation="tanh")(encoder)
    
    # Decoder
    decoder = Dense(14, activation="tanh")(encoder)
    decoder = Dense(input_dim, activation="tanh")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder