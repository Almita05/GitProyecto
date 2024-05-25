from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
import sklearn.preprocessing as pp
import pickle
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)


# Cargar los datos para obtener los rangos de escala
datos = pd.read_csv("C:/Users/almit/Downloads/DATOSPILISTO.csv", encoding='latin-1')


datos = datos.drop(['Cveestado', 'Nomestado', 'Nomddr', 'Nommunicipio', 'Nomespecie', 'Cveproducto',
                    'Nomproducto', 'Peso', 'Precio', 'Valor', 'Asacrificado', 'Año', 'Fecha_Pub_DOF',
                    'Clave ciudad', 'Nombre ciudad', 'División', 'Grupo', 'Clase', 'Subclase',
                    'Clave genérico', 'Genérico', 'Especificación', 'Cantidad', 'Unidad', 'Estatus'], axis=1)

escalador = pp.MinMaxScaler() #Se utiliza para entrenar el modelo
esc_entrenado = escalador.fit_transform(datos) #
df = pd.DataFrame(esc_entrenado, columns = datos.columns) #Remplazar el conjunto original con el escalado
df

Y = np.array(df['Precio promedio'])
datos3 = df
datos2 = df.drop('Precio promedio', axis = 1) 
X = np.array(datos2)


def load_model_from_blob():
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=storagecomputoalma;AccountKey=viqcTc6MD3m/FcjJlBhGPBKiNKcfWhWf/GO+lhtwBRGEe+EevPm8QpgWnZsBeAkG1BacriXbNfBJ+AStCOy8mA==;EndpointSuffix=core.windows.net'
    container_name = 'almacontenedor'
    blob_name = 'ModeloPI.pkl'

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

    blob_data = blob_client.download_blob().readall()
    model = pickle.loads(blob_data)
    return model


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Recupera los datos del formulario
            anio = float(request.form['anio'])
            cveddr = float(request.form['cveddr'])
            cvempio = float(request.form['cvempio'])
            cveespecie = float(request.form['cveespecie'])
            volumen = float(request.form['volumen'])
            mes = float(request.form['mes'])
            consecutivo = float(request.form['consecutivo'])
            
            X = np.array([
                anio,  # Anio
                cveddr,     # Cveddr 
                cvempio,     # Cvempio
                cveespecie,     # Cveespecie
                volumen,  # Volumen
                mes,     # Mes
                consecutivo      # Consecutivo
            ]).reshape(1,-1 )

            precios = datos[['Anio', 'Cveddr', 'Cvempio', 'Cveespecie', 'Volumen', 'Mes', 'Consecutivo']]
            escalador2 = pp.MinMaxScaler()
            escalador2 = escalador2.fit(precios)
            X = escalador2.transform(X)
            #print(X) 
            
            model = load_model_from_blob()

            pred = model.predict(X)
            print(pred) #Predicción sin desescalar
            estaturasEsc = datos3['Precio promedio'].values.reshape(-1, 1)


            escalador3 = pp.MinMaxScaler(feature_range=(datos['Precio promedio'].min(), datos['Precio promedio'].max()))
            escalador3 = escalador3.fit(estaturasEsc)
            pred = pred.reshape(1, -1)
            pred = escalador3.transform(pred)
            print(pred) #Predicción desescalada

            # Devuelve el resultado de la predicción
            return render_template('index.html', prediction_text=f'Precio: {pred[0][0]}')
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('index.html')

if __name__ == "__main__":
    app.run()