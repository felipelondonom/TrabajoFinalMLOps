import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from typing import Tuple
from ucimlrepo import fetch_ucirepo
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_dataset_from_uci():
    
   
    # El ID para tu dataset es 544
    obesity_data = fetch_ucirepo(id=544) 
    # Asumiendo que tu variable 'data' es este objeto:
    data = obesity_data
    features_df = data.data.features
    targets_df = data.data.targets
    # Concatenar las Features y el Target horizontalmente (axis=1)
    data = pd.concat([features_df, targets_df], axis=1)
    logger.info("✅ Carga de datos repositorio UCI Exitosa")

    return data
    


def feature_engineering():
    
    # Cargar el DataFrame
    data = load_dataset_from_uci()
    df = data.copy()


    # Preparación de variables con posibles decimales (por ser datos sintéticos)
    df['FCVC'] = df['FCVC'].round().astype(int)
    df['CH2O'] = df['CH2O'].round().astype(int)
    df['NCP_numeric'] = df['NCP'].round()
    
    # Eliminar duplicados completo
    df.drop_duplicates(keep='first', inplace=True)


    # 1. IMC (Índice de Masa Corporal)
    df['IMC'] = df['Weight'] / (df['Height']**2)


    # 2. Índice de Hábitos Saludables (IHS)
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


    # 3. Índice de Actividad y Sedentarismo (AIS)
    # Asignamos un peso numérico a cada medio de transporte en función de la actividad física que requiere.
    mtrans_mapping = {'Automobile': -2, 'Motorbike': -1, 'Public_Transportation': 0, 'Bike': 1, 'Walking': 2}
    df['MTRANS_Activity_Score'] = df['MTRANS'].map(mtrans_mapping)
    # Combinamos FAF (Frecuencia de Actividad Física), TUE (Tiempo usando Tecnología) y MTRANS
    # FAF y MTRANS_Activity_Score son "pro-actividad" (positivos) y TUE es "pro-sedentarismo" (negativo).
    df['AIS'] = (df['FAF'] + df['MTRANS_Activity_Score'] - df['TUE'])
    # Limpiar columna intermedia de MTRANS
    df.drop(columns=['MTRANS_Activity_Score'], inplace=True, errors='ignore')

    # 4. Generamos features finales
    data_final = df[['NObeyesdad', 'IMC', 'IHS', 'AIS', 'Gender', 'Age', 'family_history_with_overweight', 'SMOKE', 'SCC']].copy()
    logger.info("✅ Feature Engineering relizado con exito para las features IMS, IHS y AIS")


    return data_final


def encode_target_and_split_data(target_column_name: str = 'NObeyesdad', val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica Codificación Ordinal al target 'NObeyesdad', crea la columna 'y_encoded', y divide el DataFrame en Entrenamiento (80%) y Validación (20%).
    Args:
        target_column_name (str): El nombre de la columna objetivo original ('NObeyesdad').
        val_ratio (float): La proporción de los datos para el conjunto de validación (ej: 0.2).
        random_seed (int): Semilla para la reproducibilidad.
    """
    
    # 1. Copia del DataFrame para trabajar
    df_temp = feature_engineering()

    # 2. Definir el orden ordinal
    obesity_order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I','Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    
    # 3. Aplicar la Codificación Ordinal
    obesity_mapping = {level: i for i, level in enumerate(obesity_order)}
    df_temp['y_encoded'] = df_temp[target_column_name].map(obesity_mapping)
    
    # 4. Usar la columna codificada para la Estratificación
    stratify_col = df_temp['y_encoded']
    
    # 5. Dividir los datos (80% Entrenamiento / 20% Validación)
    df_train, df_val = train_test_split(
        df_temp.drop(columns=[target_column_name]), # Eliminamos el target string original
        test_size=val_ratio, 
        random_state=random_seed,
        stratify=stratify_col  
    )
    
    logger.info("✅ Codificación Ordinal aplicada y División (80/20) completada.")

    
    return df_train, df_val



def one_hot_and_ordinal_encoding():

    df_train, df_val = encode_target_and_split_data()
    # ---CODIFICACIÓN DE VARIABLES BINARIAS ---

    # 1.1 Mapeo de variables binarias
    binary_mapping_gender = {'Male': 1, 'Female': 0}
    binary_mapping_yes_no = {'yes': 1, 'no': 0}

    # 1.2 Aplicar el mapeo 
    #Train
    df_train['Gender'] = df_train['Gender'].map(binary_mapping_gender)
    df_train['family_history_with_overweight'] = df_train['family_history_with_overweight'].map(binary_mapping_yes_no)
    df_train['SMOKE'] = df_train['SMOKE'].map(binary_mapping_yes_no)
    df_train['SCC'] = df_train['SCC'].map(binary_mapping_yes_no)

    #Validation
    df_val['Gender'] = df_val['Gender'].map(binary_mapping_gender)
    df_val['family_history_with_overweight'] = df_val['family_history_with_overweight'].map(binary_mapping_yes_no)
    df_val['SMOKE'] = df_val['SMOKE'].map(binary_mapping_yes_no)
    df_val['SCC'] = df_val['SCC'].map(binary_mapping_yes_no)


 

    # 1.4 Aplicar el mapeo a la columna NObeyesdad para crear la nueva columna codificada
    df_train['NObeyesdad'] = df_train['y_encoded']
    df_val['NObeyesdad'] = df_val['y_encoded']

    # 1.5 Eliminar la columna original de string
    df_train.drop('y_encoded', axis=1, inplace=True)
    df_val.drop('y_encoded', axis=1, inplace=True)

    logger.info("✅ Codificación de variables binarias y Ordinal (0 a 6) de NObeyesdad completada.")



    return df_train, df_val



def save_processed_splits(target_col, output_path):

    df_train, df_val = one_hot_and_ordinal_encoding()

    """
    Separa los DataFrames de entrenamiento y validación en X y y, y guarda los cuatro conjuntos como archivos .pkl en la ruta especificada.
    
    Args:
        df_train (pd.DataFrame): DataFrame de entrenamiento (incluyendo el target).
        df_val (pd.DataFrame): DataFrame de validación (incluyendo el target).
        target_col (str): Nombre de la columna target codificada.
        output_path (str): Directorio donde se guardarán los archivos.
    """
    
    # 1. Crear el directorio de salida si no existe
    os.makedirs(output_path, exist_ok=True)

    # 2. Separar X y y para Entrenamiento
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    # 3. Separar X y y para Validación
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    # 4. Definir los archivos a guardar
    files_to_save = {
        "X_train.pkl": X_train,
        "y_train.pkl": y_train,
        "X_val.pkl": X_val,
        "y_val.pkl": y_val,
    }
    
    # 5. Guardar cada conjunto usando pickle
    for filename, data_set in files_to_save.items():
        filepath = os.path.join(output_path, filename)
        with open(filepath, "wb") as f_out:
            pickle.dump(data_set, f_out)
    logger.info("✅ Pipeline Completado con exito.")

# --- EJECUCIÓN ---
# El nombre del target codificado es 'NObeyesdad_encoded'
TARGET_NAME = 'NObeyesdad'
output_path= "C:/Users/Felipe Londoño M/Documents/Curso MLOps/TrabajoFinal_MLOps/src/data/proccesed"
# Ejecutar la función para separar y guardar los archivos
save_processed_splits(TARGET_NAME,output_path)