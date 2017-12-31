from django.conf.urls import url

from . import views

app_name = 'game_data'

urlpatterns = [
    #url(r'^$', views.index, name='index'),
    url(r'^$', views.import_data, name='import'),
    url(r'^create_file/', views.create_file, name='create_file'),
]