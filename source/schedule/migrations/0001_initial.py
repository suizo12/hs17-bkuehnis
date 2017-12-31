# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-10-25 16:24
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Game',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('game_id', models.IntegerField()),
                ('date', models.CharField(max_length=10)),
                ('date_uct', models.DateField()),
                ('game_code', models.CharField(max_length=45)),
                ('location', models.CharField(max_length=60)),
                ('state', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='Schedule',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField(default=0)),
                ('schedule_json', models.TextField(default=None)),
            ],
        ),
        migrations.CreateModel(
            name='Team',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tid', models.IntegerField()),
                ('short_name', models.CharField(max_length=5)),
                ('name', models.CharField(max_length=20)),
                ('city', models.CharField(max_length=20)),
            ],
        ),
        migrations.AddField(
            model_name='game',
            name='team_away',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='away_team', to='schedule.Team'),
        ),
        migrations.AddField(
            model_name='game',
            name='team_home',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='home_team', to='schedule.Team'),
        ),
    ]
