# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-11-24 14:48
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0007_auto_20171121_1845'),
    ]

    operations = [
        migrations.AddField(
            model_name='nbagame',
            name='date',
            field=models.CharField(default='', max_length=10),
            preserve_default=False,
        ),
    ]
