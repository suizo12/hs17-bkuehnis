from django.conf.urls import url

from . import views

#app_name = 'schedule'

urlpatterns = [
    #url(r'^$', views.index, name='index'),
    url(r'^$', views.index, name='index'),
    url(r'^(?P<param_date>[0-9]+-[0-9]+-[0-9]+)/$', views.gameday, name='gameday'),
    url(r'^date$', views.date, name='datetime'),
    url(r'^(?P<year>[0-9]{4})/import_schedule/$', views.import_schedule, name='import_schedule'),
    url(r'^import_games/$', views.import_game, name='import_game'),
    url(r'^about/$', views.about, name='about'),
    url(r'^schedule/$', views.schedule, name='schedule'),
]