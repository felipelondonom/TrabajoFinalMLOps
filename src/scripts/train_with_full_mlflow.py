import os
import pickle
import click
import mlflow
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nivel-sobrepeso-experiment-hpo")

def load_pickle(filename: str):
    """Carga objetos serializados (X_train, y_train, etc.) desde archivos pickle."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
@click.command()
@click.option(
    "--data_path",
    default=f"C:/Users/Felipe Londoño M/Documents/Curso MLOps/TrabajoFinal_MLOps/src/data", # Añadir 'processed' es buena práctica
    help="Ubicación donde se guardaron los datos de entrenamiento y validación (X_train.pkl, etc.)"
)
def run_optimization(data_path: str):
    # Cargar los datos. Asumimos que y_train/y_val son Series de int o float (codificadas).
    X_train = load_pickle(os.path.join(data_path, "X_train.pkl"))
    y_train = load_pickle(os.path.join(data_path, "y_train.pkl")).astype(int) # Asegurar que el target de TRAIN sea int
    X_val = load_pickle(os.path.join(data_path, "X_val.pkl"))
    y_val = load_pickle(os.path.join(data_path, "y_val.pkl")).astype(int)     # Asegurar que el target de VAL sea int

    # Función objetivo que Optuna intentará minimizar
    def objective(trial):
        with mlflow.start_run():
            # 1. Definir y registrar hiperparámetros
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }
            mlflow.log_params(params)

            # 2. Entrenar el modelo de CLASIFICACIÓN
            rf = RandomForestClassifier(**params) 
            rf.fit(X_train, y_train)

            # 3. Predicción y Cálculo de Métricas
            # Las predicciones de un Classifier son clases discretas (int) por defecto.
            y_pred = rf.predict(X_val) 
            
            # --- Métricas ---
            # Debemos asegurarnos que y_pred sea int para las métricas de clasificación
            y_pred_int = y_pred.astype(int) 

            accuracy = accuracy_score(y_val, y_pred_int)
            f1_weighted = f1_score(y_val, y_pred_int, average='weighted', zero_division=0)
            
            # 4. Registrar Métricas
            mlflow.log_metric("accuracy", accuracy) # Usado para la dirección de Optuna
            mlflow.log_metric("f1_weighted", f1_weighted)

        # Optuna requiere que la función devuelva el valor que debe minimizar
        return f1_weighted

    # Configuración y ejecución de Optuna
    sampler = optuna.samplers.TPESampler(seed=42)
    # Direccion: minimize, ya que queremos minimizar el F1
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=10)

if __name__ == '__main__':
    run_optimization()