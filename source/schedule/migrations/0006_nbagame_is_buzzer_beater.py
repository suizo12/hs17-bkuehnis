# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-11-21 16:57
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0005_nbagame'),
    ]

    operations = [
        migrations.AddField(
            model_name='nbagame',
            name='is_buzzer_beater',
            field=models.NullBooleanField(),
        ),
    ]
