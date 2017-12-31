from django.db import models


# Create your models here.

class Schedule(models.Model):
    year = models.IntegerField(default=0)
    schedule_json = models.TextField(default=None)

    def __str__(self):
        return 'Schedule'


class Team(models.Model):
    tid = models.IntegerField(primary_key=True)
    short_name = models.CharField(max_length=5)
    name = models.CharField(max_length=20)
    city = models.CharField(max_length=20)

    def __str__(self):
        return self.name + ' ' + self.city


class Game(models.Model):
    game_id = models.IntegerField()
    team_home = models.ForeignKey(Team, related_name='home_team')
    team_away = models.ForeignKey(Team, related_name='away_team')
    date = models.CharField(max_length=10)
    date_uct = models.DateField()
    game_code = models.CharField(max_length=45)
    game_code_nba = models.CharField(max_length=45)
    location = models.CharField(max_length=60)
    state = models.CharField(max_length=10)
    up_vote = models.IntegerField()
    down_vote = models.IntegerField()
    random_forest_rating = models.FloatField()


class NbaGame(models.Model):
    game_id = models.CharField(max_length=20)
    season_stage_id = models.IntegerField()
    game_code_url = models.CharField(max_length=20)
    start_time_utc = models.DateTimeField()
    h_team = models.ForeignKey(Team, related_name='h_team')
    h_score = models.IntegerField(null=True)
    h_win = models.IntegerField(null=True)
    h_loss = models.IntegerField(null=True)
    v_team = models.ForeignKey(Team, related_name='v_team')
    v_score = models.IntegerField(null=True)
    v_win = models.IntegerField(null=True)
    v_loss = models.IntegerField(null=True)
    is_buzzer_beater = models.NullBooleanField()
    date = models.CharField(max_length=10)
    downloaded_box_score = models.BooleanField(default=False)
    reddit_rating = models.IntegerField(null=True)
    wh_user_rating = models.IntegerField(null=True)
    #wikihoops_star = models.IntegerField(null=True)
    #wikihoops_rating = models.IntegerField(null=True)
    #reddit_rating = models.IntegerField(null=True)


class BoxScore:
    nba_game = models.ForeignKey(NbaGame)
    file_name = models.CharField(max_length=250)


class WikiHoops(models.Model):
    w_game_id = models.CharField(max_length=20)
    nba_game = models.ForeignKey(NbaGame, related_name='nba_game')
    game_star = models.IntegerField()
    game_rating = models.IntegerField(null=True)
    # rating_value = models.IntegerField(null=True)
    rating_up = models.IntegerField(null=True)
    rating_down = models.IntegerField(null=True)
    rating_percentage = models.CharField(max_length=5, null=True)

    def __str__(self):
        return '{0}: star[{1}] rating:[{2}]'.format(self.w_game_id, self.game_star, self.game_rating)


class NbaReddit(models.Model):
    title = models.TextField()
    ups = models.IntegerField()
    permalink = models.TextField()
    name = models.TextField()
    created = models.FloatField()
    nba_game = models.ForeignKey(NbaGame, related_name='r_nba_game', null=True)

    def __str__(self):
        return '{3}: {0} - {1}: [{2}]'.format(self.created, self.ups, self.title, self.nba_game_id)