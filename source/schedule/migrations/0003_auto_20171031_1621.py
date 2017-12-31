# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-10-31 15:21
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0002_auto_20171027_1222'),
    ]

    operations = [
        migrations.AddField(
            model_name='game',
            name='down_vote',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='game',
            name='random_forest_rating',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='game',
            name='up_vote',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]