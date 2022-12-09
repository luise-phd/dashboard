# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import render

from . import models
from django.core.files.storage import FileSystemStorage
from django.urls import reverse

import os
import threading
import psycopg2
import psycopg2.extras
import pyproj
import rasterio
import numpy as np
import pandas as pd
from unipath import Path
from datetime import datetime

from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from scipy.stats import loguniform

@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]
        
        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))

        if load_template == 'analizar_datos.html':
            context = analizar_datos(request)
        elif load_template == 'analisis_histo.html':
            context = analisis_histo(request)
        elif load_template == 'algoritmos.html':
            context = algoritmos(request)
        elif load_template == 'ver_diagrama.html':
            context = ver_diagrama(request)
        elif load_template == 'archivos.html':
            context = archivos(request)
        elif load_template == 'predicciones_histo.html':
            context = predicciones_histo(request)
        else:
            context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:
        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))



def predicciones_histo(request):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM predicciones_histo WHERE usuario='"+request.user.username+"' ORDER BY fecha DESC")
        datos = cur.fetchall()
    
    context = {"datos": datos, "segment": 'predicciones_histo'}
    return context


def descargar_csv_rad_solar(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM predicciones_histo WHERE idprh = '"+str(id)+"'")
        datos = cur.fetchall()
        file_path = os.path.join(Path(__file__).parent.parent.parent, r'data\\'+ datos[0][0])
        print(file_path)
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + datos[0][0]
            return response


def eliminar_csv_rad_solar(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM predicciones_histo WHERE idprh='"+str(id)+"'")
        nom_archivo = cur.fetchone()

        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("DELETE FROM predicciones_histo WHERE idprh='"+str(id)+"'")

        cur3 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur3.execute("SELECT * FROM predicciones_histo WHERE usuario = '"+request.user.username+"' ORDER BY fecha DESC")
        datos = cur3.fetchall()

        fs = FileSystemStorage()
        filename = fs.delete(nom_archivo[0][0])

    context = {"datos": datos}
    return render(request, '../templates/home/predicciones_histo.html', context=context)


# Distancia de la tierra al sol
def distancia( dia_juliano ):
    dia_rad = 2. * np.pi * (dia_juliano - 1.) / 365.
    return 1.00011 + 0.034221 * np.cos(dia_rad) + 0.00128 * np.sin(dia_rad) + 0.000719 * np.cos(2. * dia_rad) + 0.000077 * np.sin(2. * dia_rad)

# Declinación Solar
def declinacion_solar( dia_juliano ):
    dia_rad = 2. * np.pi * (dia_juliano - 1.) / 365.
    return (0.006918 - (0.399912 * np.cos(dia_rad)) + (0.070257 * np.sin(dia_rad)) - (0.006758 * np.cos(2. * dia_rad)) + (0.00097 * np.sin(2. * dia_rad)) - (0.002697 * np.cos(3. * dia_rad)) + (0.00148 * np.sin(3. * dia_rad))) * (180. / np.pi)

# Ecuación del Tiempo
def e_t( dia_juliano ):
    dia_rad = 2. * np.pi * (dia_juliano - 1.) / 365.
    return (0.000075 + 0.001868 * np.cos(dia_rad) - 0.032077 * np.sin(dia_rad) - 0.014615 * np.cos(2. * dia_rad) - 0.04089 * np.sin(2. * dia_rad)) * 229.18

# Tiempo solar verdadero
def tiempo_solar_verdadero( dia_juliano, lon, hlc ):
    meridiano_ref = -75.
    return (((lon - meridiano_ref) * 4. + e_t( dia_juliano )) / 60.) + hlc

# Ángulo cenital
def cos_zenit(lat, lon, dia_juliano, hlc):
    decl = declinacion_solar(dia_juliano)
    angulo_horario = (12 - tiempo_solar_verdadero(dia_juliano, lon, hlc)) * 15.
    return np.sin(np.radians(lat)) * np.sin(np.radians(decl)) + np.cos(np.radians(lat)) * np.cos(np.radians(decl)) * np.cos(np.radians(angulo_horario))


def procesar_imgs(request):
    nom = "".join(request.POST.getlist('nombre')).strip().replace(' ', '-')
    lat = float(request.POST['latitud'].replace(',', '.'))
    lon = float(request.POST['longitud'].replace(',', '.'))
    year = request.POST['ann']
    print(year)
    
    files = os.listdir('imgs/' + year + '/')
    images = []
    for elem in files:
        if (elem.find('.tiff') == -1 and elem != 'GeoTiff'):
            images.append(elem)

    df_ccal = pd.read_csv('data/Coeficiente de calibración.csv', sep=';')
    df_fcorr = pd.read_csv('data/Factor de corrección GOES.csv', sep=';')

    csol = 1367

    df = pd.DataFrame({'anno': [], 'mes': [], 'dia': [], 'hora': [], 'ndig': [], 'reflectancia': []})
    df['rmin'] = 0
    df['rmax'] = 0
    df['nc'] = 0
    df['N'] = 0
    df['Hext'] = 0
    df['brilloSolarC'] = 0
    df['dj'] = 0

    band = 0
    x = 0
    for i in images:
        if (i.find('.tiff') == -1):
            goes_vis = rasterio.open('imgs/' + year + '/' + i)

            goes_vis2 = goes_vis.read(1).astype('float64')
            ratio_goes = np.where((goes_vis2 - 1) == 0., 0, (goes_vis2) / (goes_vis2 - 1))
            ratio_goesImage = rasterio.open('imgs/' + year + '/GeoTiff/'+ i +'.tiff', 'w', driver='Gtiff',
                                            width = goes_vis.width,
                                            height = goes_vis.height,
                                            count = 1,
                                            crs = {'init': 'EPSG:4326'},
                                            transform = goes_vis.transform,
                                            dtype='float64')
            ratio_goesImage.write(ratio_goes, 1)
            ratio_goesImage.close()

            # Extracción de valores
            localname = 'imgs/' + year + '/GeoTiff/'+ i +'.tiff'
            with rasterio.open(localname) as src: src.profile
            row, col = src.index(lon, lat) # spatial --> image coordinates
            try:
                nd = goes_vis2[row, col]
            except IndexError:
                print('Error. Coordenadas por fuera de la imágen')
                band = -1
                break
            print(f'Nivel digital = {nd:.2f}')

            # Coeficiente de calibración
            k = 0.001160

            l = i.split('_')

            ann = int(l[3][:4])
            mes = int(l[3][4:6])
            fcorr = df_fcorr[(df_fcorr['anno'] == ann) & (df_fcorr['mes'] == mes)]
            
            # Factor de correción
            C = float(fcorr['goes13'])
            print("Factor de corrección", ann, mes, C)

            # Reflectancia nominal
            Rprev = k * (nd - 29)

            # Reflectancia posterior
            Rpost = C * Rprev

            # Día Juliano
            dt = datetime.strptime(l[3], "%Y%m%d")
            dj = float(dt.strftime("%j"))
            print('Día Juliano:', dj)

            # Hora local
            hlc = float(l[4][:2]) - 4
            print('Hora local:', hlc)
            
            r = distancia(dj)
            Oz = cos_zenit(lat, lon, dj, hlc)
            Rp = Rpost * np.power(r, 2) / np.cos(Oz)
            
            cos_w = -(np.tan(np.radians(lat))) * np.tan(np.radians(declinacion_solar(dj)))
            W = np.arccos(cos_w) * 180 / np.pi
            # Duración astronómica del día
            N = (2 / 15) * W
            # Radiación solar diaria en el tope de la atmosfera
            Hext = (24 / np.pi) * csol * distancia(dj) * ((np.cos(lat * np.pi / 180) * np.cos(declinacion_solar(dj) * np.pi / 180) * np.sin(np.arccos(cos_w))) + ((2 * np.pi / 360) * (np.arccos(cos_w) * 180 /  np.pi) * np.sin(declinacion_solar(dj) * np.pi / 180) * np.sin(lat * np.pi / 180)))

            df.loc[x] = [l[3][:4], l[3][4:6], l[3][6:8], hlc, nd, Rp, 0, 0, 0, N, Hext, 0, dj]
            x += 1
            print(x)

            print('-------------------------------------------------')

    if(band != -1):
        t = threading.Thread(target = realizar_prediccion, args = (df, request, nom, year, lat, lon))
        t.start()
    
    context = {"datos": band}    
    return render(request, '../templates/home/radiacion_solar.html', context=context)


def realizar_prediccion(df, request, nom, year, lat, lon):
    # Reflectancia mínima y máxima por hora, asociada a condiciones de cielo despejado y cubierto
    df_r_min_max = df.groupby(['hora'], as_index=False).agg(rmin = ('reflectancia','min'), rmax=('reflectancia','max'))

    l1 = df.values.tolist()
    l2 = df_r_min_max.values.tolist()

    for elem in l1:
        for elem2 in l2:
            # Reflectancia mínima
            if (elem[3] == elem2[0]):
                elem[6] = elem2[1]
                elem[7] = elem2[2] * 0.8 # multiplicar por 80%, segun Laguarda

    for elem in l1:
        elem[8] = (elem[5] - elem[6]) / (elem[7] - elem[6])
        # Índice de nubosidad (0-1)
        if (elem[8] >= 1.):
            elem[8] = 1.
        if (elem[8] < 0.):
            elem[8] = 0.
        # Brillo solar calculado
        elem[11] = 1 - elem[8]

    df_nc = pd.DataFrame(l1, columns=['anno', 'mes', 'dia', 'hora', 'ndig', 'reflectancia', 'rmin', 'rmax', 'nc', 'N', 'Hext', 'brilloSolarC', 'dj'])

    df_nc = df_nc.apply(pd.to_numeric)
    df_nc.sort_values(by=['anno', 'mes', 'dia', 'hora'], inplace=True, ignore_index=True)

    # Cargar datos
    df_acmocoa_2012 = pd.read_csv('data/ds_train/DS-ACMocoa-2012.csv', sep=';')
    df_acmocoa_2013 = pd.read_csv('data/ds_train/DS-ACMocoa-2013.csv', sep=';')
    df_acmocoa_2014 = pd.read_csv('data/ds_train/DS-ACMocoa-2014.csv', sep=';')
    df_acmocoa = pd.concat([df_acmocoa_2012, df_acmocoa_2013, df_acmocoa_2014], ignore_index=True, sort=False)
    df_acmocoa.insert(0, 'Estacion', 'ACMocoa', allow_duplicates=False)
    df_acmocoa = df_acmocoa.drop(df_acmocoa[(df_acmocoa['RadSolar'] <= 0)].index)

    df_el_pepino_2012 = pd.read_csv('data/ds_train/DS-El-Pepino-2012.csv', sep=';')
    df_el_pepino_2013 = pd.read_csv('data/ds_train/DS-El-Pepino-2013.csv', sep=';')
    df_el_pepino_2014 = pd.read_csv('data/ds_train/DS-El-Pepino-2014.csv', sep=';')
    df_el_pepino = pd.concat([df_el_pepino_2012, df_el_pepino_2013, df_el_pepino_2014], ignore_index=True, sort=False)
    df_el_pepino.insert(0, 'Estacion', 'El Pepino', allow_duplicates=False)
    df_el_pepino = df_el_pepino.drop(df_el_pepino[(df_el_pepino['RadSolar'] <= 0)].index)

    df_putumayo = pd.concat([df_acmocoa, df_el_pepino], ignore_index=True, sort=False)

    df_epatia_2012 = pd.read_csv('data/ds_train/DS-CEPatia-2012.csv', sep=';')
    df_epatia_2013 = pd.read_csv('data/ds_train/DS-CEPatia-2013.csv', sep=';')
    df_epatia_2014 = pd.read_csv('data/ds_train/DS-CEPatia-2014.csv', sep=';')
    df_epatia = pd.concat([df_epatia_2012, df_epatia_2013, df_epatia_2014], ignore_index=True, sort=False)
    df_epatia.insert(0, 'Estacion', 'Estrecho Patía', allow_duplicates=False)
    df_epatia = df_epatia.drop(df_epatia[(df_epatia['RadSolar'] <= 0)].index)

    df_aguapi = pd.read_csv('data/ds_train/DS-CGuapi-2014.csv', sep=';')
    df_aguapi.insert(0, 'Estacion', 'Aeropuerto Guapi', allow_duplicates=False)
    df_aguapi = df_epatia.drop(df_epatia[(df_epatia['RadSolar'] <= 0)].index)

    df_cauca = pd.concat([df_epatia, df_aguapi], ignore_index=True, sort=False)

    df_florencia_2012 = pd.read_csv('data/ds_train/DS-Florencia-2012.csv', sep=';')
    df_florencia_2013 = pd.read_csv('data/ds_train/DS-Florencia-2013.csv', sep=';')
    df_florencia_2014 = pd.read_csv('data/ds_train/DS-Florencia-2014.csv', sep=';')
    df_florencia = pd.concat([df_florencia_2012, df_florencia_2013, df_florencia_2014], ignore_index=True, sort=False)
    df_florencia.insert(0, 'Estacion', 'Florencia', allow_duplicates=False)
    df_florencia = df_florencia.drop(df_florencia[(df_florencia['RadSolar'] <= 0)].index)

    df_fmacagual_2012 = pd.read_csv('data/ds_train/DS-FMacagual-2012.csv', sep=';')
    df_fmacagual_2014 = pd.read_csv('data/ds_train/DS-FMacagual-2014.csv', sep=';')
    df_fmacagual = pd.concat([df_fmacagual_2012, df_fmacagual_2014], ignore_index=True, sort=False)
    df_fmacagual.insert(0, 'Estacion', 'Macagual', allow_duplicates=False)
    df_fmacagual = df_fmacagual.drop(df_fmacagual[(df_fmacagual['RadSolar'] <= 0)].index)

    df_caqueta = pd.concat([df_florencia, df_fmacagual], ignore_index=True, sort=False)

    # Concatenar sets de datos
    # df3 = pd.concat([df_putumayo, df_cauca, df_caqueta], ignore_index=True, sort=False)
    df3 = pd.concat([df_cauca], ignore_index=True, sort=False)

    # Preparando los datos para el análisis
    X = df3.drop(['Estacion', 'ann', 'RadSolar'], axis=1)
    y = df3['RadSolar']

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=100)

    # Random Forest. Ajuste de hiperparámetros con busqueda aleatoria
    rf = RandomForestRegressor()

    # Número de árboles en un bosque aleatorio
    n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num = 10)]
    # Número de características a considerar en cada división
    max_features = ['sqrt']
    # Número máximo de niveles en el árbol
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_split = [2, 5, 10, 15, 20]
    # Número mínimo de muestras requeridas en cada nodo hoja
    min_samples_leaf = [1, 2, 4, 6]
    # Método de selección de muestras para entrenar cada árbol 
    bootstrap = [True, False]
    grid = dict(n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap)

    print("[INFO] grid searching over the hyperparameters...")
    cvFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=rf,
                                    n_jobs=-1,
                                    cv=cvFold,
                                    param_distributions=grid,
                                    scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train.ravel())

    # extract the best model and evaluate it
    print("[INFO] evaluating...")
    bestModel = searchResults.best_estimator_

    print("R2: {:.2f}".format(bestModel.score(X_train, y_train)))
    print("R2 (Test): {:.2f}".format(bestModel.score(X_test, y_test)))

    print(bestModel)
    print("[INFO] grid search best parameters: {}".format(searchResults.best_params_))

    print("")

    y_train_hat = bestModel.predict(X_train)
    y_test_hat = bestModel.predict(X_test)
    print("Entrenamiento: ", r2_score(y_train, y_train_hat))
    print("Prueba: ", r2_score(y_test, y_test_hat))

    df_final = df_nc.drop(['anno', 'ndig', 'rmin', 'rmax', 'brilloSolarC', 'dj'], axis=1)

    X_final = pd.DataFrame(scaler_X.fit_transform(df_final), columns=df_final.columns)
    y_final = bestModel.predict(X_final)

    X_final_no_scaler = pd.DataFrame(scaler_X.inverse_transform(X_final))
    y_final_hat = pd.DataFrame(scaler_y.inverse_transform(y_final.reshape(-1, 1)))

    df_final['RadSolar'] = y_final_hat

    # Exportar dataset
    df_final.sort_values(by=['mes', 'dia', 'hora'], inplace=True, ignore_index=True)
    nom_archivo = 'DS-RadSolar-' + request.user.username + '-' + nom + '-' + year + '.csv'
    df_final.to_csv('data/' + nom_archivo, sep=';', index=False)

    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT COUNT(*) FROM predicciones_histo WHERE archivo='"+nom_archivo+"'")
        result = cur.fetchall()
        if result[0][0] > 0:
            cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
            cur2.execute("UPDATE predicciones_histo SET fecha = current_timestamp WHERE archivo = '"+nom_archivo+"'")
        else:
            cur3 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
            cur3.execute("""
                INSERT INTO predicciones_histo (lugar, lat, lon, ann, fecha, usuario, archivo)
                VALUES ('"""+nom+"""', '"""+str(lat)+"""', '"""+str(lon)+"""', '"""+year+"""', current_timestamp, '"""+request.user.username+"""', '"""+nom_archivo+"""')""")
            
        # cur4 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        # cur4.execute("SELECT * FROM predicciones_histo WHERE usuario='"+request.user.username+"'")
        # datos = cur4.fetchall()


def ver_diagrama(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT var_objetivo, al.algoritmo, res_train, res_test, resultados FROM analisis_histo ah, algoritmos al WHERE idana='"+str(id)+"' AND ah.idalg = al.idalg")
        datos = cur.fetchall()
    
    context = {"datos": datos}
    return render(request, '../templates/home/ver_diagrama.html', context=context)


def eliminar_analisis_histo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("DELETE FROM analisis_histo WHERE idana='"+str(id)+"'")

        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("SELECT a.idana, a.usuario, a.variables, a.var_objetivo, al.algoritmo, a.res_train, a.res_test, a.fecha, a.archivo FROM analisis_histo a, algoritmos al WHERE a.usuario = '"+request.user.username+"' AND a.idalg = al.idalg ORDER BY a.idana DESC")
        datos = cur2.fetchall()
    
    context = {"datos": datos}
    return render(request, '../templates/home/analisis_histo.html', context=context)


def realizar_analisis(request):
    arch_sel=""
    var_obj=""
    columns=""
    alg_sel=""
    resultados=""
    if request.method == 'POST' and request.POST['arch_sel'] != "":
        archivos = list_archivos(request)
        arch_sel = request.POST['arch_sel']
        var_obj = request.POST.getlist('var_obj')
        predictores = request.POST.getlist('predictores[]')

        alg_sel = request.POST['alg_sel']

        if '.csv' in arch_sel:
            df = pd.read_csv('data/'+request.user.username+'--'+arch_sel, sep=';')
        elif '.xls' in arch_sel or '.xlsx' in arch_sel:
            df = pd.read_excel('data/'+request.user.username+'--'+arch_sel, engine='openpyxl')
        elif '.txt' in arch_sel:
            df = pd.read_csv('data/'+request.user.username+'--'+arch_sel, delimiter='\t')

        columns = df.columns

        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT algoritmo FROM algoritmos ORDER BY algoritmo")
        algoritmos = cur.fetchall()

        if request.POST.getlist('var_obj') != []:
            try:
                # X = df.drop(var_obj, axis=1)
                X = df[predictores]
                y = df[var_obj]

                scaler_X = RobustScaler()
                scaler_y = RobustScaler()

                X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
                y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=100)

                band = 0

                algs = {
                    'Regresión Lineal Múltiple': "LinearRegression()",
                    'Regresión Polinomial': "PolynomialFeatures(degree=2)",
                    'Árboles de Regresión': "DecisionTreeRegressor(max_depth=12, min_samples_leaf=50)",
                    'Regresión con soporte vectorial (SVR)': "SVR(kernel='rbf', gamma='auto')",
                    'Bosques Aleatorios para Regresión (RF)': "RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=6)",
                    'Redes Neuronales Artificiales':
                        "MLPRegressor(solver='adam', alpha=1e-6, hidden_layer_sizes=(100, 75, 50, 25), n_iter_no_change=20,activation='relu', learning_rate='constant', max_iter=10000, warm_start=True)",
                    'AdaBoost Regressor': "AdaBoostRegressor(n_estimators=50, learning_rate=0.3)",
                    'Gradient Boosting Regressor': "GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=4)",
                    'XGBoost Regressor':
                        "XGBRegressor(n_estimators=50, n_jobs=-1, learning_rate=.5, max_depth=2, colsample_bytree=1, subsample=.6, objective='reg:squarederror', verbosity=1)"
                }

                if alg_sel != 'Regresión Polinomial':
                    regr = eval(algs[alg_sel])

                    regr.fit(X_train, y_train.ravel())
                    y_train_hat = regr.predict(X_train)
                    y_test_hat = regr.predict(X_test)
                else:
                    poly = eval(algs[alg_sel])
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.fit_transform(X_test)

                    reg_poly = LinearRegression().fit(X_train_poly, y_train)
                    y_train_hat = reg_poly.predict(X_train_poly)
                    y_test_hat = reg_poly.predict(X_test_poly)

                res_train = round(r2_score(y_train, y_train_hat), 5)
                res_test = round(r2_score(y_test, y_test_hat), 5)

                # if alg_sel == 'Bosques Aleatorios para Regresión (RF)':
                #     # Solo se grafica el último grupo de predicciones
                #     kf = KFold(n_splits=5, shuffle=True)
                #     rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=6)

                #     resultados_y_train = []
                #     resultados_y_test = []

                #     for train_index, test_index in kf.split(X):
                #         X_train, X_test = X.loc[train_index,], X.loc[test_index,]
                #         y_train, y_test = y[train_index], y[test_index]
                #         rf.fit(X_train, y_train.ravel())
                #         y_train_hat = rf.predict(X_train)
                #         y_test_hat = rf.predict(X_test)
                #         resultados_y_test.append(r2_score(y_test, y_test_hat))
                #         resultados_y_train.append(r2_score(y_train, y_train_hat))

                y_test_list = ""
                for valores in scaler_y.inverse_transform(y_test.reshape(-1, 1)):
                    for v in valores:
                        y_test_list += str(round(v, 0)) + ";"

                y_test_hat_list = ""
                for valores in scaler_y.inverse_transform(y_test_hat.reshape(-1, 1)):
                    for v in valores:
                        y_test_hat_list += str(round(v, 0)) + ";"

                resultados = [res_train, res_test, y_test_list, y_test_hat_list]

                vars=""
                for col in X.columns:
                    vars += col + ", "

                if connection() != None:
                    cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                    cur.execute("SELECT idalg FROM algoritmos WHERE algoritmo = %s", (alg_sel, ))
                    idalg = cur.fetchall()
                    
                    cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                    cur2.execute("""
                        INSERT INTO analisis_histo (usuario, variables, var_objetivo, idalg, res_train, res_test, fecha, resultados, archivo)
                        VALUES ('"""+request.user.username+"""', '"""+vars[:-2]+"""', '"""+var_obj[0]+"""', %s, %s, %s, current_timestamp, %s, %s)
                    """, (idalg[0][0], res_train, res_test, y_test_list+'||'+y_test_hat_list, arch_sel))

            except ValueError:
                print('La variable objetivo seleccionada no tiene registros')
            except KeyError:
                print("La variable objetivo seleccionada no tiene Ejes")
    else:
        archivos = list_archivos(request)
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT algoritmo FROM algoritmos ORDER BY algoritmo")
        algoritmos = cur.fetchall()

    context = {
        "archivos": archivos,
        "arch_sel": arch_sel,
        "var_obj": var_obj,
        "columns": columns,
        "alg_sel": alg_sel,
        "resultados": resultados,
        "algoritmos": algoritmos
    }
    if alg_sel == '':
        return render(request, '../templates/home/analizar_datos.html', context=context)
    else:
        return render(request, '../templates/home/analisis_histo.html', context=analisis_histo(request))


def analisis_histo(request):
    datos=""
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT a.idana, a.usuario, a.variables, a.var_objetivo, al.algoritmo, a.res_train, a.res_test, a.fecha, a.archivo FROM analisis_histo a, algoritmos al WHERE a.usuario = '"+request.user.username+"' AND a.idalg = al.idalg ORDER BY a.idana DESC")
        datos = cur.fetchall()

    context = {"datos": datos, "segment": 'analisis_histo'}
    return context


def list_archivos(request):
    datos=""
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM archivos WHERE usuario = '"+request.user.username+"' ORDER BY archivo")
        datos = cur.fetchall()
    return datos


def edit_algoritmo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM algoritmos WHERE idalg='"+str(id)+"'")
        datos = cur.fetchall()
    
    context = {"datos": datos}
    return render(request, '../templates/home/new_algoritmo.html', context=context)


def delete_algoritmo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("DELETE FROM algoritmos WHERE idalg='"+str(id)+"'")

        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("SELECT * FROM algoritmos ORDER BY algoritmo")
        datos = cur2.fetchall()
    
    context = {"datos": datos}
    return render(request, '../templates/home/algoritmos.html', context=context)


def new_algoritmo(request):
    return render(request, '../templates/home/new_algoritmo.html', context={})


def algoritmos(request):
    algoritmo = "".join(request.POST.getlist('algoritmo'))
    
    try:
        idalg = request.POST.getlist('editar')[0]
    
        if connection() != None:
            if request.POST.getlist('editar')[0] == "":
                cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                cur.execute("INSERT INTO algoritmos (algoritmo) VALUES ('"+algoritmo+"')")
            else:
                cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                cur.execute("UPDATE algoritmos SET algoritmo = '"+algoritmo+"' WHERE idalg = '"+idalg+"'")
    except:
        print('No se ejecuto una operación de adicionar o de editar')

    if connection() != None:
        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("SELECT * FROM algoritmos ORDER BY algoritmo")
        datos = cur2.fetchall()
    
    context = {"datos": datos, "segment": 'algoritmos'}
    return context


def archivos(request):
    if connection() != None:
        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("SELECT * FROM archivos WHERE usuario = '"+request.user.username+"' ORDER BY fecha DESC")
        datos = cur2.fetchall()
    
    context = {"datos": datos, "segment": 'archivos'}
    return context


def descargar_archivo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM archivos WHERE idarch = "+str(id)+" ORDER BY archivo")
        datos = cur.fetchall()
        nom_arch = request.user.username + '--' + ''.join(datos[0])
        file_path = os.path.join(Path(__file__).parent.parent.parent, r'data\\'+ nom_arch)
        print(file_path)
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + ''.join(datos[0])
            return response


def eliminar_archivo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM archivos WHERE idarch='"+str(id)+"'")
        nom_archivo = cur.fetchone()

        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("DELETE FROM archivos WHERE idarch='"+str(id)+"'")

        cur3 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur3.execute("SELECT * FROM archivos WHERE usuario = '"+request.user.username+"' ORDER BY fecha DESC")
        datos = cur3.fetchall()

        fs = FileSystemStorage()
        filename = fs.delete(request.user.username + '--' + "".join(nom_archivo[0]))

    context = {"datos": datos}
    return render(request, '../templates/home/archivos.html', context=context)


def abrir_archivo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM archivos al WHERE idarch='"+str(id)+"'")
        datos = cur.fetchall()
        try:
            nom_archivo = "".join(datos[0])
            ruta_arc = 'data/'+request.user.username+'--'+nom_archivo
        except:
            cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
            cur.execute("SELECT archivo FROM predicciones_histo al WHERE idprh='"+str(id)+"'")
            datos = cur.fetchall()
            nom_archivo = "".join(datos[0])
            ruta_arc = 'data/'+nom_archivo

        if '.csv' in nom_archivo:
            df = pd.read_csv(ruta_arc, sep=';')
        elif '.xls' in nom_archivo or '.xlsx' in nom_archivo:
            df = pd.read_excel(ruta_arc, engine='openpyxl')
        elif '.txt' in nom_archivo:
            df = pd.read_csv(ruta_arc, delimiter='\t')
        datos = df.values.tolist()
        columns = df.columns

    context = {"datos": datos, "nom_arch": nom_archivo, "columns": columns}
    return render(request, '../templates/home/abrir_archivo.html', context=context)


def analizar_datos(request):
    archivos = list_archivos(request)

    cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
    cur.execute("SELECT algoritmo FROM algoritmos ORDER BY algoritmo")
    algoritmos = cur.fetchall()

    context = {"archivos": archivos, "algoritmos": algoritmos, "segment": 'analizar'}
    return context


def subir_csv(request):
    datos=""
    cant_reg=0
    no_num=0

    if request.method == 'POST' and request.FILES['myFile']:
        usuario = request.POST['usuario']
        myFile = request.FILES['myFile']

        fs = FileSystemStorage()
        if not fs.exists(usuario + '--' + myFile.name):
            filename = fs.save(usuario + '--' + myFile.name, myFile)
            uploaded_file_url = fs.url(filename)

            if '.csv' in filename:
                df = pd.read_csv('data/'+filename, sep=';')
            elif '.xls' in filename or '.xlsx' in filename:
                df = pd.read_excel('data/'+filename, engine='openpyxl')
            elif '.txt' in filename:
                df = pd.read_csv('data/'+filename, delimiter='\t')
            datos = df.head(100).values.tolist()
            columns = df.columns
            if(df.shape[0] == 0):
                cant_reg = -1
            for col in columns:
                if(df[col].dtype == "object"):
                    no_num = -1

            if connection() != None and cant_reg != -1 and no_num != -1:
                cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                cur.execute("""
                    INSERT INTO archivos (usuario, archivo, fecha)
                    VALUES ('"""+usuario+"""', '"""+filename.split(usuario + '--')[-1]+"""', current_timestamp)
                """)
        else:
            filename = fs.delete(usuario + '--' + myFile.name)
            filename = fs.save(usuario + '--' + myFile.name, myFile)
            if '.csv' in myFile.name:
                df = pd.read_csv('data/'+usuario+'--'+myFile.name, sep=';')
            elif '.xls' in myFile.name or '.xlsx' in myFile.name:
                df = pd.read_excel('data/'+usuario+'--'+myFile.name, engine='openpyxl')
            elif '.txt' in myFile.name:
                df = pd.read_csv('data/'+usuario+'--'+myFile.name, delimiter='\t')
            datos = df.head(100).values.tolist()
            columns = df.columns
            if(df.shape[0] == 0):
                cant_reg = -1
            for col in columns:
                if(df[col].dtype == "object"):
                    no_num = -1

    context = {
        "datos": datos,
        "nom_arch": myFile.name+" (Primeras 100 filas...)",
        "columns": columns,
        "cant_reg": cant_reg,
         "no_num": no_num
    }
    return render(request, '../templates/home/index.html', context=context)


def connection():
    hostname = 'localhost'
    database = 'doctorado'
    username = 'postgres'
    pwd = 'postgres'
    port_id = 5432

    conn = None

    try:
        conn = psycopg2.connect (
            host = hostname,
            database = database,
            user = username,
            password = pwd,
            port = port_id
        )
        conn.set_session(autocommit=True)
        
    except Exception as error:
        print('Mensaje de error: ', error)
    
    return conn