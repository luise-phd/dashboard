# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views

urlpatterns = [
    # The home page
    path('', views.index, name='home'),

    path('subir_csv/', views.subir_csv, name='subir_csv'),
    path('realizar_analisis/', views.realizar_analisis, name='realizar_analisis'),

    path('diagrama_disp_results/<int:id>/', views.diagrama_disp_results, name='diagrama_disp_results'),
    path('eliminar_analisis_histo/<int:id>/', views.eliminar_analisis_histo, name='eliminar_analisis_histo'),

    path('algoritmos/', views.algoritmos, name='algoritmos'),
    path('new_algoritmo/', views.new_algoritmo, name='new_algoritmo'),
    path('delete_algoritmo/<int:id>/', views.delete_algoritmo, name='delete_algoritmo'),
    path('edit_algoritmo/<int:id>/', views.edit_algoritmo, name='edit_algoritmo'),

    path('archivos/', views.archivos, name='archivos'),
    path('eliminar_archivo/<int:id>/', views.eliminar_archivo, name='eliminar_archivo'),
    path('abrir_archivo/<int:id>/', views.abrir_archivo, name='abrir_archivo'),
    path('descargar_archivo/<int:id>/', views.descargar_archivo, name='descargar_archivo'),
    path('descargar_archivo_base/', views.descargar_archivo_base, name='descargar_archivo_base'),

    path('procesar_imgs/', views.procesar_imgs, name='procesar_imgs'),
    path('diagrama_rad_solar/<int:id>/', views.diagrama_rad_solar, name='diagrama_rad_solar'),
    path('descargar_csv_rad_solar/<int:id>/', views.descargar_csv_rad_solar, name='descargar_csv_rad_solar'),
    path('eliminar_csv_rad_solar/<int:id>/', views.eliminar_csv_rad_solar, name='eliminar_csv_rad_solar'),

    path('visualizar_dashboard/', views.visualizar_dashboard, name='visualizar_dashboard'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages')
]