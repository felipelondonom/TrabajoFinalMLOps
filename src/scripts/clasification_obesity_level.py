#!/usr/bin/env python
# coding: utf-8


import os
import pickle
import logging
from pathlib import Path


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier 
import mlflow
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from ucimlrepo import fetch_ucirepo 



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test connection
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        mlflow.set_experiment("obesity-level-experiment-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
setup_mlflow()

@task(
    name="Load UCI Data", 
    description="Descarga el dataset de niveles de obesidad del repositorio UCI, consolida las features y targets, y elimina duplicados.", 
    retries=3, 
    retry_delay_seconds=10
)
def read_dataframe(datase_id: int) -> pd.DataFrame:
    """
    Carga el dataset 'Estimation of Obesity Levels' de UCI, elimina registros duplicados
    y reporta estadísticas básicas.

    Args:
        datase_id (int): El ID numérico del dataset en el repositorio UCI (ej: 544).

    Returns:
        pd.DataFrame: DataFrame consolidado con todas las features y el target.
    """
    logger = get_run_logger()
    
    dataset_id_uci = datase_id 
 
    
    logger.info(f"Cargando dataset ID: {dataset_id_uci}")
    try:
        data = fetch_ucirepo(id=datase_id) 
        features_df = data.data.features
        targets_df = data.data.targets

        # Concatenar Features y Target (axis=1)
        df = pd.concat([features_df, targets_df], axis=1)

        # Eliminar duplicados (Optimización de eficiencia)
        initial_count = len(df)
        df.drop_duplicates(keep='first', inplace=True)
        dropped_count = initial_count - len(df)
        
        logger.info(f"✅ Cargado exitosamente {len(df)} registros (eliminados {dropped_count} duplicados).")
    
    except Exception as e:
        logger.error(f"Fallo al cargar datos desde UCI ID {dataset_id_uci}: {e}")
        # En Prefect, es buena práctica relanzar la excepción para que la tarea falle
        raise 


    # --- Creación de Artefacto Prefect ---
    summary_data = [
        ["Total samples", len(df)],
        ["Average Age", f"{df['Age'].mean():.2f} years"],
        ["Min Age", f"{df['Age'].min():.2f} years"],
        ["Max Age", f"{df['Age'].max():.2f} years"]
    ]

    create_table_artifact(
        key=f"data-summary",
        table=summary_data,
        description=f"Resumen estadístico básico del DataFrame recién cargado."
    )

    return df



@task(
      name="Encode Target and Split Data", 
      description="Aplica Codificación Ordinal al target 'NObeyesdad' y divide el DataFrame en conjuntos de Entrenamiento y Validación (80/20) con Estratificación.",
      log_prints=True # Hace que logger.info se imprima directamente)
)
def split_data(df: pd.DataFrame, target_column_name: str = 'NObeyesdad', val_ratio: float = 0.2, random_seed: int = 42) -> pd.DataFrame:
    """
    Realiza la Codificación Ordinal del target y divide el DataFrame en conjuntos de 
    entrenamiento y validación (80/20) usando estratificación.

    Args:
        df (pd.DataFrame): DataFrame de entrada (post-Feature Engineering).
        target_column_name (str): Nombre de la columna objetivo original ('NObeyesdad').
        val_ratio (float): Proporción de los datos para el conjunto de validación (ej: 0.2).
        random_seed (int): Semilla para la reproducibilidad del split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: df_train (80%) y df_val (20%), ambos con 
                                           la columna target codificada ('y_encoded').
    """

    logger = get_run_logger()

    # --- 1. Definir el orden ordinal (Esquema de Clasificación) ---
    obesity_order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I','Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    
    # 2. Aplicar la Codificación Ordinal
    obesity_mapping = {level: i for i, level in enumerate(obesity_order)}
    df['y_encoded'] = df[target_column_name].map(obesity_mapping)
    
    # 3. Preparar la Columna de Estratificación y Limpieza
    stratify_col = df['y_encoded']

    # Eliminamos el target string original ANTES de hacer el split
    df_to_split = df.drop(columns=[target_column_name]) 
    
    # 4. Dividir los datos (80% Entrenamiento / 20% Validación)
    df_train, df_val = train_test_split(
        df_to_split, 
        test_size=val_ratio, 
        random_state=random_seed,
        stratify=stratify_col
    )

    logger.info(f"✅ División completada. Train shape: {df_train.shape}, Val shape: {df_val.shape}")
    
    return df_train, df_val



@task(
    name="Feature Engineering)", 
    description="Calcula el IMC, el Índice de Hábitos Saludables (IHS) y el Índice de Actividad y Sedentarismo (AIS), condensando múltiples variables en 3 métricas robustas.",
    log_prints=True
)
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la ingeniería de características clave (IMC, IHS, AIS) a partir de los datos brutos.
    
    Optimiza el cálculo del IHS y AIS eliminando columnas temporales innecesarias.

    Args:
        df (pd.DataFrame): DataFrame de entrada (post-carga y post-limpieza de duplicados).

    Returns:
        pd.DataFrame: DataFrame con las features originales seleccionadas y las 3 nuevas features esenciales (IMC, IHS, AIS).
    """
    logger = get_run_logger()
    

    # Preparación de variables con posibles decimales (por ser datos sintéticos)
    df['FCVC'] = df['FCVC'].round().astype(int)
    df['CH2O'] = df['CH2O'].round().astype(int)
    df['NCP_numeric'] = df['NCP'].round()


    # 1. CÁLCULO DE IMC (Riesgo Físico Primario)
    df['IMC'] = df['Weight'] / (df['Height']**2)

    # 2. CÁLCULO DE IHS (Índice de Hábitos Saludables)
    df['FAVC_Score'] = df['FAVC'].map({'no': 1, 'yes': -1})
    df['FCVC_Score'] = df['FCVC'].map({1: -1, 2: 0, 3: 1})
    df['NCP_Score'] = 1 - abs(df['NCP_numeric'] - 3)
    df['CAEC_Score'] = df['CAEC'].map({'No': 2, 'Sometimes': 1, 'Frequently': -1, 'Always': -2})
    df['CH2O_Score'] = df['CH2O'].map({1: -1, 2: 0, 3: 1})
    df['CALC_Score'] = df['CALC'].map({'no': 2, 'Sometimes': 1, 'Frequently': -1, 'Always': -2})
    # Suma de todas las puntuaciones para el IHS
    score_cols = ['FAVC_Score', 'FCVC_Score', 'NCP_Score','CAEC_Score', 'CH2O_Score', 'CALC_Score']
    df['IHS'] = df[score_cols].sum(axis=1)
    df.drop(columns=score_cols + ['NCP_numeric'], inplace=True, errors='ignore')


    # 3. CÁLCULO DE AIS (Índice de Actividad y Sedentarismo)
    # Asignamos un peso numérico a cada medio de transporte en función de la actividad física que requiere.
    mtrans_mapping = {'Automobile': -2, 'Motorbike': -1, 'Public_Transportation': 0, 'Bike': 1, 'Walking': 2}
    df['MTRANS_Activity_Score'] = df['MTRANS'].map(mtrans_mapping)
    # Combinamos FAF (Frecuencia de Actividad Física), TUE (Tiempo usando Tecnología) y MTRANS
    # FAF y MTRANS_Activity_Score son "pro-actividad" (positivos) y TUE es "pro-sedentarismo" (negativo).
    df['AIS'] = (df['FAF'] + df['MTRANS_Activity_Score'] - df['TUE'])
    # Limpiar columna intermedia de MTRANS
    df.drop(columns=['MTRANS_Activity_Score'], inplace=True, errors='ignore')

    # 4. Generamos features finales
    df = df[['NObeyesdad', 'IMC', 'IHS', 'AIS', 'Gender', 'Age', 'family_history_with_overweight', 'SMOKE', 'SCC']].copy()

    logger.info(f"✅ Creación de Features IHS, IMC, AIS completada.")
  
    return df


@task(
    name="One Hot Encoding", 
    description="Aplica Codificación Binaria (0/1) a las features categóricas predictoras (Gender, family_history_with_overweight, SMOKE, SCC).",
    log_prints=True
)
def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Codificación Binaria (mapeo fijo 0/1) a las features predictoras.

    Args:
        df (pd.DataFrame): DataFrame de entrada (post-Feature Engineering).

    Returns:
        pd.DataFrame: DataFrame con las features binarias convertidas a enteros (0 ó 1).
    """
    # 1. Definir los mapeos
    binary_mapping_gender = {'Male': 1, 'Female': 0}
    binary_mapping_yes_no = {'yes': 1, 'no': 0}

    # 2. Columnas a codificar (Las que permanecen en el DataFrame final)
    binary_cols_yes_no = ['family_history_with_overweight', 'SMOKE', 'SCC']

    # 3. Aplicar mapeo
    df['Gender'] = df['Gender'].map(binary_mapping_gender).astype(int)
    
    # Aplicar mapeo a las columnas YES/NO de forma eficiente
    for col in binary_cols_yes_no:
        df[col] = df[col].map(binary_mapping_yes_no).astype(int)


    logger.info("✅ Codificación de variables binarias completada (Gender, family_history_with_overweight, SMOKE, SCC).")
    return df





@task(
    name="Train Model Classifier", 
    description="Entrena el modelo Random Forest Classifier con hiperparámetros de baseline, usando TRAIN y VAL, y registra todos los resultados en MLflow y Prefect Artifacts.",
    log_prints=True
)
def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> str:
    """
    Entrena el modelo Random Forest Classifier para la clasificación multiclase de obesidad.
    
    Asegura que el target sea entero (int) y registra métricas clave (F1-weighted, Accuracy, RMSE) 
    en MLflow para seguimiento.

    Args:
        X_train (pd.DataFrame): Conjunto de entrenamiento (features).
        y_train (pd.Series): Target de entrenamiento (codificado como int).
        X_val (pd.DataFrame): Conjunto de validación (features).
        y_val (pd.Series): Target de validación (codificado como int).
    
    Returns:
        str: El ID de la ejecución (Run ID) de MLflow.
    """
    logger = get_run_logger()
    
    # 0. Configuración del Entorno de Guardado
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    # CRÍTICO: Asegurar que las etiquetas sean enteras (0, 1, 2... 6) antes de usar con el clasificador
    y_train_int = y_train.astype(int)
    y_val_int = y_val.astype(int)
    
    logger.info(f"Iniciando entrenamiento con {X_train.shape[0]} muestras y {X_train.shape[1]} features.")

    with mlflow.start_run() as run:
        # 1. Definir Hiperparámetros para CLASIFICACIÓN (ejemplo de HPO de Optuna)
        best_params = {
            # Se optimizan los hiperparámetros comunes de Random Forest
            'n_estimators': 22,
            'max_depth': 11,              
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        mlflow.log_params(best_params)

        # 2. Entrenar el modelo de CLASIFICACIÓN
        # Usamos RandomForestClassifier
        model = RandomForestClassifier(**best_params)

        # El entrenamiento es directo con el clasificador
        model.fit(X_train, y_train_int)

        # 3. Predicción y Cálculo de Métricas de CLASIFICACIÓN
        y_pred = model.predict(X_val) 
        y_pred_int = y_pred.astype(int)
        
        # Métricas
        accuracy = accuracy_score(y_val_int, y_pred_int)
        f1_weighted = f1_score(y_val_int, y_pred_int, average='weighted', zero_division=0)
        


        # 4. Registrar Métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # 5. Registro de Artefactos (Modelo)
        try:
            # log_model registra el modelo entrenado y los metadatos
            mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
            logger.info("✅ Modelo Random Forest registrado en MLflow.")
        except Exception as e:
            logger.warning(f"Fallo al registrar el modelo en MLflow: {e}")

        # 6. Crear Artefactos de Prefect para visualización
        performance_data = [
            ["F1 Ponderado", f"{f1_weighted:.4f}"],
            ["Precisión (Accuracy)", f"{accuracy:.4f}"],
            ["Estimadores", best_params['n_estimators']],
            ["Profundidad Máxima", best_params['max_depth']],
            ["MLflow Run ID", run.info.run_id]
        ]

        create_table_artifact(
            key="model-performance",
            table=performance_data,
            description=f"RF Classifier: F1 Ponderado en Validación: {f1_weighted:.4f}"
        )
        
        # Crear markdown artifact con resumen
        markdown_content = f"""
        # Resumen de Entrenamiento - Random Forest Baseline

        ## Rendimiento (Conjunto de Validación)
        - **F1 Ponderado**: {f1_weighted:.4f}
        - **Precisión (Accuracy)**: {accuracy:.4f}
        - **MLflow Run ID**: {run.info.run_id}
        """
        create_markdown_artifact(
            key="training-summary",
            markdown=markdown_content,
            description="Resumen detallado del entrenamiento del modelo Random Forest."
        )

        return run.info.run_id
    


    
@flow(
    name="Obesity Level Classification Pipeline", 
    description="Orquesta la carga, Feature Engineering, preprocesamiento (codificación y escalado), y entrenamiento de un clasificador Random Forest, registrando los resultados en MLflow.",
    log_prints=True
)
def level_obesity_flow(id:int) -> str:
    """
    Orquesta el pipeline de MLOps para predecir los niveles de obesidad.

    Args:
        id (int): ID numérico del dataset en el repositorio UCI (ej: 544).
    
    Returns:
        str: El ID de la ejecución (Run ID) de MLflow para el modelo entrenado.
    """
    logger = get_run_logger()
    
    # --- 1. CARGA DE DATOS ---
    # La tarea read_dataframe también debe manejar la limpieza de duplicados.
    df_raw = read_dataframe(id)

    # --- 2. FEATURE ENGINEERING (IMC, IHS, AIS) ---
    # CRÍTICO: Las Features se crean UNA SOLA VEZ en el DataFrame completo.
    df_processed = create_features(df_raw)
    
    # --- 3. CODIFICACIÓN ORDINAL DEL TARGET Y SPLIT ---
    # split_data aplica Codificación Ordinal al target y divide el DF.
    # df_train y df_val contienen ya las Features (IMC, IHS, AIS) y el target ('y_encoded').
    df_train, df_val = split_data(df_processed)

    # --- 4. PREPROCESAMIENTO FINAL (Codificación Binaria y Escalado) ---
    # Esta tarea aplica el mapeo 0/1 a las columnas como Gender y el StandardScaler
    # a las continuas (IMC, AIS, Age) siguiendo la regla de NO DATA LEAKAGE.
    df_train = one_hot_encoding(df_train)
    df_val = one_hot_encoding(df_val)
    # --- 5. SEPARACIÓN FINAL X/Y ---
    # El target codificado debe ser el que usamos.
    TARGET_ENCODED_COL = "y_encoded" 
    
    # Separar X e Y de entrenamiento
    X_train = df_train.drop(columns=[TARGET_ENCODED_COL])
    y_train = df_train[TARGET_ENCODED_COL]

    # Separar X e Y de validación
    X_val = df_val.drop(columns=[TARGET_ENCODED_COL])
    y_val = df_val[TARGET_ENCODED_COL]
    
    logger.info(f"Datos listos para entrenar: X_train={X_train.shape}, X_val={X_val.shape}")

    # --- 6. ENTRENAMIENTO Y REGISTRO ---
    run_id = train_model(X_train, y_train, X_val, y_val)

    # --- 7. ARTEFACTO RESUMEN DEL PIPELINE ---
    pipeline_summary = f"""
    # Resumen de Ejecución del Pipeline MLOps

    ## Datos
    - **Muestras de Entrenamiento**: {len(y_train):,}
    - **Muestras de Validación**: {len(y_val):,}
    - **Características**: {X_train.shape[1]} (Incluye IMC, IHS, AIS)

    ## Resultados
    - **Modelo**: Random Forest Classifier (Baseline)
    - **MLflow Run ID**: {run_id}

    ## Siguientes Pasos
    1. Revisar el rendimiento del modelo en MLflow UI: http://localhost:5000
    """

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=pipeline_summary,
        description="Resumen completo de la ejecución del pipeline"
    )

    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Orquesta el pipeline de clasificación de niveles de obesidad con Prefect y MLflow.')
    parser.add_argument('--id', type=int, default=544, help='ID numérico del dataset de obesidad de UCI (default: 544).')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (sobrescribe la configuración por defecto).')
    args = parser.parse_args()

    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        setup_mlflow()

    try:
        # Run the flow
        run_id = level_obesity_flow(544)
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 MLflow run_id: {run_id}")
        print(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
        logger.info("ID de la ejecución guardado en prefect_run_id.txt")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


