# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-29 15:14
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0018_auto_20171214_1235'),
    ]

    operations = [
        migrations.AddField(
            model_name='nbagame',
            name='reddit_rating',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='nbagame',
            name='wh_user_rating',
            field=models.IntegerField(null=True),
        ),
    ]